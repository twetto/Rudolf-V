#!/usr/bin/env python3
"""Score dense stereo on a ZED recording against the SDK's NEURAL_LIGHT depth, next to OpenCV DIS.

    python3 zed_record.py extract walk.svo2 walk --every 10        # on the Orin
    cargo run --release --example dense_stereo_eval -- walk/rect --out ours > ours/frames.jsonl
    cargo run --release --example dense_stereo_eval -- walk/raw --out raw > raw/frames.jsonl  # optional
    python3 -I zed_eval.py walk ours [--raw raw] [--png DIR] [--png-every 20]

The SDK's depth (rectified left camera) is sampled on the depth maps' grid (level 1: every other
pixel). DIS runs as the robot pipeline's fallback does (mr-neuro-nav: preset FAST, spatial
propagation, half-resolution images, left -> right), its horizontal flow taken as disparity.
Our raw-image depth is warped onto the rectified grid through the SDK's rectification rotation,
found from SIFT matches between raw and rectified left images.

Prints, per method:
  - coverage of the SDK's pixels, log-range errors overall and by the SDK's range, and the
    shares nearer / farther than the SDK by over 25 %;
  - the obstacle band (mr-neuro-nav's risk band: the top 45 % of the image): nearest range per
    column too near / too far by over 10 % or missing, where the SDK has 20 % of the column;
    risk per bin (share of valid band pixels with Z under 0.5 m / 1 m, 9 bins over 67 deg)
    against the SDK's; blind bins (under 5 % valid: the dog stops).

NEURAL_LIGHT is a learned network that fills textureless walls from context: a reference, not
truth. Missing depth there is by design (issue #5: no invented depth).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import yaml

RANGES = [0.3, 0.5, 1.0, 2.0, 3.5, 5.0, np.inf]  # SDK range bins, metres
EDGES = np.linspace(-2, 2, 4001)  # signed log-range error histogram, 0.001 bins
BAND = 0.45  # mr-neuro-nav reflex: risk_y_max_frac
N_BINS, FOV = 9, np.radians(67.0)  # mr-neuro-nav configs/quadruped.yaml
SAFETY = (0.5, 1.0)
BLIND = 0.05  # blind_stop_coverage


def camchain(path):
    c = yaml.safe_load(open(path))
    def cam(e):
        fx, fy, cx, cy = e["intrinsics"]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), np.array(e["distortion_coeffs"], float)
    K0, d0 = cam(c["cam0"])
    return K0, d0, np.array(c["cam1"]["T_cn_cnm1"], float)


def rays(K, xs, ys):
    """Unit rays of rectified (pinhole) pixels."""
    b = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones_like(xs, dtype=float)], -1)
    return b / np.linalg.norm(b, axis=-1, keepdims=True)


def undistort(pts, K, d):
    """Normalised coordinates of raw pixels, OpenCV rational model, by Newton (cv2.undistortPoints
    stops after 5 fixed-point iterations)."""
    k1, k2, p1, p2, k3, k4, k5, k6 = d
    t = (pts - K[:2, 2]) / np.diag(K)[:2]
    def distort(p):
        x, y = p[:, 0], p[:, 1]
        r2 = x * x + y * y
        rad = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        return np.stack([x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                         y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y], -1)
    p, h = t.copy(), 1e-7
    for _ in range(20):
        f = distort(p) - t
        jx = (distort(p + [h, 0]) - distort(p - [h, 0])) / (2 * h)
        jy = (distort(p + [0, h]) - distort(p - [0, h])) / (2 * h)
        det = jx[:, 0] * jy[:, 1] - jy[:, 0] * jx[:, 1]
        p -= np.stack([jy[:, 1] * f[:, 0] - jy[:, 0] * f[:, 1], jx[:, 0] * f[:, 1] - jx[:, 1] * f[:, 0]], -1) / det[:, None]
    return p


def rectification(src, stems, K_rect, K_raw, d_raw, n=12):
    """Rotation taking raw left-camera rays to rectified ones, from SIFT matches."""
    sift = cv2.SIFT_create(4000)
    A, B = [], []
    for s in stems[:: max(1, len(stems) // n)][:n]:
        a = cv2.imread(os.path.join(src, "raw/left", s + ".png"), 0)
        b = cv2.imread(os.path.join(src, "rect/left", s + ".png"), 0)
        ka, da = sift.detectAndCompute(a, None)
        kb, db = sift.detectAndCompute(b, None)
        for m, m2 in cv2.BFMatcher().knnMatch(da, db, k=2):
            if m.distance < 0.7 * m2.distance:
                A.append(ka[m.queryIdx].pt)
                B.append(kb[m.trainIdx].pt)
    na = undistort(np.array(A), K_raw, d_raw)
    ba = np.c_[na, np.ones(len(na))]
    ba /= np.linalg.norm(ba, axis=1, keepdims=True)
    B = np.array(B)
    bb = rays(K_rect, B[:, 0], B[:, 1])
    keep = np.ones(len(ba), bool)
    for _ in range(10):  # Kabsch, then drop the worst matches
        U, _, Vt = np.linalg.svd(ba[keep].T @ bb[keep])
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        R = (U @ D @ Vt).T
        err = np.degrees(np.arccos(np.clip(np.sum((ba @ R.T) * bb, 1), -1, 1)))
        keep = err < max(np.percentile(err[keep], 90), 0.05)
    px = np.median(err[keep]) * np.radians(1) * K_rect[0, 0]
    print(f"raw -> rectified rotation: {np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))):.3f} deg "
          f"from {keep.sum()} matches, residual median {px:.2f} px")
    return R


class Score:
    """One method's errors against the SDK, accumulated over frames (as histograms: frames x
    pixels is too many to keep)."""

    def __init__(self):
        nr = len(RANGES) - 1
        self.sdk = np.zeros(nr)
        self.hist = np.zeros((nr, len(EDGES) - 1))
        self.px = self.all_px = 0
        self.col = []  # nearest-per-column log errors; NaN: no estimate in that column
        self.risk = {d: [] for d in SAFETY}
        self.blind, self.sdk_blind = [], []

    def add(self, est, sdk, z_of_logr, col_bin):
        self.px += np.isfinite(est).sum()
        self.all_px += est.size
        r = np.exp(sdk)
        has = np.isfinite(est) & np.isfinite(sdk)
        for i in range(len(RANGES) - 1):
            sel = (r >= RANGES[i]) & (r < RANGES[i + 1])
            self.sdk[i] += sel.sum()
            e = np.clip((est - sdk)[sel & has], EDGES[0] + 1e-6, EDGES[-1] - 1e-6)
            self.hist[i] += np.histogram(e, EDGES)[0]
        rows = int(BAND * est.shape[0])
        gb, eb = sdk[:rows], est[:rows]
        cols = np.isfinite(gb).mean(0) >= 0.2
        with np.errstate(all="ignore"):
            g_near = np.nanmin(np.where(np.isfinite(gb), gb, np.inf), 0)
            e_near = np.nanmin(np.where(np.isfinite(eb), eb, np.inf), 0)
        e_near[~np.isfinite(e_near)] = np.nan
        self.col.append((e_near - g_near)[cols])
        # mr-neuro-nav compute_depth_risk_with_coverage (binary), on Z.
        zg, ze = z_of_logr(gb), z_of_logr(eb)
        bins = col_bin[col_bin >= 0]
        total = rows * np.bincount(bins, minlength=N_BINS)
        def risk(z, d):
            valid = np.isfinite(z)
            n = np.bincount(bins, np.sum(valid, 0)[col_bin >= 0], N_BINS)
            near = np.bincount(bins, np.sum(valid & (z < d), 0)[col_bin >= 0], N_BINS)
            return np.where(n > 0, near / np.maximum(n, 1), 0.0), n / total
        for d in SAFETY:
            (rg, cov_g), (re, cov) = risk(zg, d), risk(ze, d)
            self.risk[d].append(np.abs(re - rg))
        self.blind.append(cov < BLIND)
        self.sdk_blind.append(cov_g < BLIND)

    @staticmethod
    def _median(h, edges):
        c = np.cumsum(h)
        if c[-1] == 0:
            return np.nan
        i = np.searchsorted(c, 0.5 * c[-1])
        return 0.5 * (edges[i] + edges[i + 1])

    def stats(self, rows=None):
        h = self.hist.sum(0) if rows is None else self.hist[rows]
        mid = len(h) // 2
        a = h[mid:] + h[:mid][::-1]  # |error|, bins EDGES[mid:]
        n = max(h.sum(), 1)
        at = lambda x: np.searchsorted(EDGES, x)
        sdk = self.sdk.sum() if rows is None else self.sdk[rows]
        return {"cov": h.sum() / max(sdk, 1), "abs_med": self._median(a, EDGES[mid:]),
                "sgn": self._median(h, EDGES),
                "b10": a[at(np.log(1.1)) - mid:].sum() / n, "b25": a[at(np.log(1.25)) - mid:].sum() / n,
                "near25": h[: at(-np.log(1.25))].sum() / n, "far25": h[at(np.log(1.25)):].sum() / n}


def report(name, s):
    st = s.stats()
    print(f"\n{name}: {100 * s.px / s.all_px:.1f}% of pixels have depth, {100 * st['cov']:.1f}% of the SDK's")
    print(f"  |log-range error| median {100 * st['abs_med']:.2f}%, signed {100 * st['sgn']:+.2f}%, "
          f">10% {100 * st['b10']:.1f}%, >25% {100 * st['b25']:.1f}% "
          f"(nearer {100 * st['near25']:.1f}%, farther {100 * st['far25']:.1f}%)")
    print(f"  {'SDK range':>12s} {'SDK px':>9s} {'coverage':>9s} {'median':>8s} {'signed':>8s} {'>10%':>7s} {'>25%':>7s}")
    for i in range(len(RANGES) - 1):
        b = s.stats(i)
        print(f"  {RANGES[i]:5.1f}-{RANGES[i + 1]:<5.1f}m {int(s.sdk[i]):9d} {100 * b['cov']:8.1f}% "
              f"{100 * b['abs_med']:7.2f}% {100 * b['sgn']:+7.2f}% {100 * b['b10']:6.1f}% {100 * b['b25']:6.1f}%")
    c = np.concatenate(s.col)
    miss = ~np.isfinite(c)
    cc = c[~miss]
    print(f"  band, nearest per column ({len(c)} columns): median |err| {100 * np.median(np.abs(cc)):.2f}%, "
          f"too near {100 * np.mean(cc < -np.log(1.1)) * (1 - miss.mean()):.1f}%, "
          f"too far {100 * np.mean(cc > np.log(1.1)) * (1 - miss.mean()):.1f}%, no depth {100 * miss.mean():.1f}%")
    risk = ", ".join(f"Z<{d} m {np.mean(np.concatenate(s.risk[d])):.3f}" for d in SAFETY)
    print(f"  band, risk per bin vs SDK, mean |diff|: {risk}")
    blind = lambda b: f"{100 * np.mean(np.concatenate(b)):.1f}% of bins, {100 * np.mean([x.any() for x in b]):.1f}% of frames"
    print(f"  band, blind (under {100 * BLIND:.0f}% valid): {blind(s.blind)}; the SDK's {blind(s.sdk_blind)}")


def colour(logr):
    t = np.clip((logr - np.log(0.3)) / (np.log(6.0) - np.log(0.3)), 0, 1)
    img = np.stack([255 * t, 255 * (1 - np.abs(2 * t - 1)), 255 * (1 - t)], -1)  # BGR: near red, far blue
    img[~np.isfinite(logr)] = 0
    return img.astype(np.uint8)


def err_colour(e):
    t = np.clip(e / np.log(1.2), -1, 1)  # +-20 %
    img = np.stack([255 * np.clip(t, 0, 1), np.zeros_like(t), 255 * np.clip(-t, 0, 1)], -1)  # red: nearer
    img[~np.isfinite(e)] = 40
    return img.astype(np.uint8)


def label(img, text, band=False):
    img = img.copy()
    if band:
        cv2.line(img, (0, int(BAND * img.shape[0])), (img.shape[1], int(BAND * img.shape[0])), (255, 255, 255), 1)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("src", help="zed_record.py extract output (with rect/ and raw/)")
    ap.add_argument("ours", help="dense_stereo_eval --out on src/rect, with frames.jsonl")
    ap.add_argument("--raw", help="dense_stereo_eval --out on src/raw")
    ap.add_argument("--png", help="write comparison images here")
    ap.add_argument("--png-every", type=int, default=20)
    args = ap.parse_args()

    K, _, T = camchain(os.path.join(args.src, "rect/camchain.yaml"))
    baseline = -T[0, 3]
    frames = [json.loads(l) for l in open(os.path.join(args.ours, "frames.jsonl"))]
    stems = [f["stem"] for f in frames]
    W, H = frames[0]["width"], frames[0]["height"]
    s = 2 ** frames[0]["level"]
    ys, xs = np.mgrid[0:H, 0:W]
    n_grid = 1 / rays(K, xs * s, ys * s)[..., 2]  # range / Z on our grid
    n_dis = 1 / rays(K, xs * s + 0.5 * (s - 1), ys * s + 0.5 * (s - 1))[..., 2]  # INTER_AREA pixel centres
    z_of = lambda logr: np.exp(logr) / n_grid[: logr.shape[0]]
    col_bin = np.arctan2(np.arange(W) * s - K[0, 2], K[0, 0])
    col_bin = np.searchsorted(np.linspace(-FOV / 2, FOV / 2, N_BINS + 1), col_bin, side="right") - 1
    col_bin[(col_bin < 0) | (col_bin >= N_BINS)] = -1

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    dis.setUseSpatialPropagation(True)

    if args.raw:
        K_raw, d_raw, _ = camchain(os.path.join(args.src, "raw/camchain.yaml"))
        R = rectification(args.src, stems, K, K_raw, d_raw)
        # Where each rectified grid pixel's ray lands on the raw grid (same centre: same range).
        uv, _ = cv2.projectPoints(rays(K, xs * s, ys * s).reshape(-1, 3) @ R, np.zeros(3), np.zeros(3), K_raw, d_raw)
        uv = uv.reshape(H, W, 2) / s
        RX, RY = np.round(uv[..., 0]).astype(int), np.round(uv[..., 1]).astype(int)
        inside = (RX >= 0) & (RY >= 0) & (RX < W) & (RY < H)

    scores = {"ours": Score(), "DIS": Score()}
    if os.path.exists(os.path.join(args.ours, stems[0] + ".src")):
        scores["ours, finest level only"] = Score()
        scores["ours, coarse fallback only"] = Score()
    if args.raw:
        scores["ours, raw images"] = Score()
    if args.png:
        os.makedirs(args.png, exist_ok=True)
    dis_ms = []
    for k, stem in enumerate(stems):
        z = cv2.imread(os.path.join(args.src, "rect/depth", stem + ".png"), -1)[::s, ::s][:H, :W] / 1000.0
        sdk = np.where(z > 0, np.log(np.maximum(z, 1e-3) * n_grid), np.nan).astype(np.float32)
        ours = np.fromfile(os.path.join(args.ours, stem + ".logr"), "<f4").reshape(H, W)
        left = cv2.imread(os.path.join(args.src, "rect/left", stem + ".png"), 0)
        right = cv2.imread(os.path.join(args.src, "rect/right", stem + ".png"), 0)
        t0 = time.perf_counter()
        a = cv2.resize(left, (left.shape[1] // s, left.shape[0] // s), interpolation=cv2.INTER_AREA)
        b = cv2.resize(right, (right.shape[1] // s, right.shape[0] // s), interpolation=cv2.INTER_AREA)
        flow = dis.calc(a, b, None)[:H, :W]
        dis_ms.append((time.perf_counter() - t0) * 1e3)
        disp = -flow[..., 0] * s  # full-resolution pixels
        with np.errstate(all="ignore"):
            d_logr = np.where(disp > 0, np.log(K[0, 0] * baseline / disp * n_dis), np.nan).astype(np.float32)
        d_logr[(d_logr < np.log(0.25)) | (d_logr > np.log(30))] = np.nan
        est = {"ours": ours, "DIS": d_logr}
        src_path = os.path.join(args.ours, stem + ".src")
        if os.path.exists(src_path):  # source level per pixel: split measured / coarse fallback
            src = np.fromfile(src_path, np.uint8).reshape(H, W)
            for name, sel in (("ours, finest level only", src == frames[0]["level"]),
                              ("ours, coarse fallback only", (src != 255) & (src != frames[0]["level"]))):
                est[name] = np.where(sel, ours, np.nan).astype(np.float32)
        if args.raw:
            raw = np.fromfile(os.path.join(args.raw, stem + ".logr"), "<f4").reshape(H, W)
            warped = np.full((H, W), np.nan, np.float32)
            warped[inside] = raw[RY[inside], RX[inside]]
            est["ours, raw images"] = warped
        for name, e in est.items():
            scores[name].add(e, sdk, z_of, col_bin)
        if args.png and k % args.png_every == 0:
            grey = lambda im: cv2.cvtColor(im[::s, ::s], cv2.COLOR_GRAY2BGR)
            pct = lambda e: f"{100 * np.isfinite(e).mean():.0f}% px"
            rows = [np.hstack([label(grey(left), "rectified left", True), label(grey(right), "rectified right")]),
                    np.hstack([label(colour(sdk), f"SDK NEURAL_LIGHT ({pct(sdk)})", True),
                               label(colour(ours), f"ours ({pct(ours)})", True)]),
                    np.hstack([label(colour(d_logr), f"OpenCV DIS fast ({pct(d_logr)})", True),
                               label(err_colour(ours - sdk), "ours vs SDK", True)])]
            if args.raw:
                rows.append(np.hstack([label(colour(raw), f"ours on raw images ({pct(raw)})"),
                                       label(err_colour(d_logr - sdk), "DIS vs SDK", True)]))
            legend = np.full((30, 2 * W, 3), 255, np.uint8)
            cv2.putText(legend, "range 0.3 m red .. 6 m blue, black = none; error red = nearer than SDK, blue = "
                        "farther, full at 20 %; line: obstacle band", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(args.png, stem + ".png"), np.vstack(rows + [legend]))

    print(f"{len(stems)} frames, {W}x{H} grid; DIS fast {np.median(dis_ms):.1f} ms median on this machine")
    for name, sc in scores.items():
        report(name, sc)


if __name__ == "__main__":
    main()
