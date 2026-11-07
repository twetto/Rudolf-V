#!/usr/bin/env python3
"""Score dense stereo on a ZED recording against the SDK's NEURAL_LIGHT depth, next to OpenCV DIS and SGBM.

    python3 zed_record.py extract walk.svo2 walk --every 10        # on the Orin
    cargo run --release --example dense_stereo_eval -- walk/rect --out ours > ours/frames.jsonl
    cargo run --release --example dense_stereo_eval -- walk/raw --out raw > raw/frames.jsonl  # optional
    python3 -I zed_eval.py walk ours [--raw raw] [--png DIR] [--png-every 20]
                                     [--dis fast medium fast:0] [--sgbm 2 2:hh 8:sgbm:3]
                                     [--threads 1] [--on-ours]

The SDK's depth (rectified left camera) is sampled on the depth maps' grid (level 1: every other
pixel). DIS runs as the robot pipeline's fallback does (mr-neuro-nav: preset FAST, spatial
propagation, half-resolution images, left -> right), its horizontal flow taken as disparity and
sampled on the same grid.
--dis scores other presets too, PRESET:SCALE also setting the finest scale DIS computes (FAST: 2,
i.e. a quarter of its half-resolution input; ours computes every pixel of that grid, scale 0).
DIS is timed on its calc() alone, the downsampling to half resolution apart (ours includes its
pyramid). Every frame starts without a flow: DIS takes any flow of the right size it is passed
as its initial estimate.
--sgbm scores OpenCV's semi-global matching, SCALE[:MODE[:BLOCK]]: images downsampled SCALE times
(a divisor of the grid's, which samples the disparity), MODE sgbm (5 paths), hh (8), 3way or hh4,
BLOCK the matching window (default 5). Its settings follow OpenCV's stereo_match sample (P1, P2 =
8, 32 x BLOCK^2, uniqueness 10 %, left-right check within 1 px) with ours' range (0.25 m and
farther) and speckle filter (400 full-resolution pixels, 1 px of disparity). Both images are
padded on the left by the disparity range (zeros) and the result cropped back: SGBM leaves its
first numDisparities columns (37 % of a ZED image) empty otherwise. Pixels whose match falls left
of the right image are dropped (ours' test; with repeated-border padding 6 % of SGBM's depth
matched there). Timed on compute() alone, padding included.
--nn scores a stereo network's disparity from nn_infer.py (its TensorRT engine on the Orin),
DIR[:CONF]: DIR holds meta.json (the network's image size; the full image was resized to it) and
<stem>.disp (float16 disparity in the network's pixels), with :CONF also <stem>.conf, pixels under
CONF dropped. Each grid pixel takes the nearest network pixel's disparity.
--on-ours also scores each DIS, SGBM and network configuration on only the pixels where ours has depth: the
same pixels, so errors compare like for like.
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
EDGES = np.linspace(-2, 2, 20001)  # signed log-range error histogram, 0.0002 bins
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
    print(f"  |log-range error| median {100 * st['abs_med']:.3f}%, signed {100 * st['sgn']:+.2f}%, "
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
    ap.add_argument("--dis", nargs="+", default=["fast"], metavar="PRESET[:SCALE]",
                    help="DIS configurations to score: ultrafast, fast or medium, optionally :finest scale")
    ap.add_argument("--sgbm", nargs="*", default=[], metavar="SCALE[:MODE[:BLOCK]]",
                    help="SGBM configurations to score: image downsampling, sgbm/hh/3way/hh4, window")
    ap.add_argument("--nn", nargs="*", default=[], metavar="DIR[:CONF]",
                    help="network disparity maps (nn_infer.py), optionally with a confidence threshold")
    ap.add_argument("--threads", type=int, help="OpenCV threads (default: OpenCV's choice)")
    ap.add_argument("--on-ours", action="store_true", help="also score DIS where ours has depth")
    args = ap.parse_args()
    if args.threads:
        cv2.setNumThreads(args.threads)

    K, _, T = camchain(os.path.join(args.src, "rect/camchain.yaml"))
    baseline = -T[0, 3]
    frames = [json.loads(l) for l in open(os.path.join(args.ours, "frames.jsonl"))]
    stems = [f["stem"] for f in frames]
    W, H = frames[0]["width"], frames[0]["height"]
    s = 2 ** frames[0]["level"]
    ys, xs = np.mgrid[0:H, 0:W]
    n_grid = 1 / rays(K, xs * s, ys * s)[..., 2]  # range / Z on our grid
    # DIS sees half-resolution images as the pipeline does (full resolution for a level-0 grid),
    # and is sampled every `step` pixels on our grid; its pixel centres, after INTER_AREA:
    sd = min(s, 2)
    step = s // sd
    n_dis = 1 / rays(K, xs * s + 0.5 * (sd - 1), ys * s + 0.5 * (sd - 1))[..., 2]
    z_of = lambda logr: np.exp(logr) / n_grid[: logr.shape[0]]
    col_bin = np.arctan2(np.arange(W) * s - K[0, 2], K[0, 0])
    col_bin = np.searchsorted(np.linspace(-FOV / 2, FOV / 2, N_BINS + 1), col_bin, side="right") - 1
    col_bin[(col_bin < 0) | (col_bin >= N_BINS)] = -1

    presets = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
               "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}
    dis = {}  # name -> DIS instance
    for spec in args.dis:
        preset, _, scale = spec.partition(":")
        d = cv2.DISOpticalFlow_create(presets[preset])
        d.setUseSpatialPropagation(True)
        name = f"DIS {preset}"
        if scale:
            d.setFinestScale(int(scale))
            name += f", finest scale {scale}"
        dis[name] = d
    first_dis = next(iter(dis))

    modes = {"sgbm": cv2.STEREO_SGBM_MODE_SGBM, "hh": cv2.STEREO_SGBM_MODE_HH,
             "3way": cv2.STEREO_SGBM_MODE_SGBM_3WAY, "hh4": cv2.STEREO_SGBM_MODE_HH4}
    sgbm = {}  # name -> (downsampling, matcher, disparity range)
    for spec in args.sgbm:
        parts = spec.split(":")
        sc, mode = int(parts[0]), parts[1] if len(parts) > 1 else "sgbm"
        bs = int(parts[2]) if len(parts) > 2 else 5
        assert s % sc == 0, f"--sgbm {spec}: the downsampling must divide the grid's ({s})"
        nd = int(np.ceil(K[0, 0] * baseline / (0.25 * sc) / 16)) * 16
        m = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=bs, P1=8 * bs * bs,
                                  P2=32 * bs * bs, disp12MaxDiff=1, preFilterCap=63, uniquenessRatio=10,
                                  speckleWindowSize=max(1, 400 // (sc * sc)), speckleRange=1, mode=modes[mode])
        sgbm[f"SGBM {mode}, 1/{sc} resolution, block {bs}"] = (sc, m, nd)
    first_sgbm = next(iter(sgbm), None)

    nets = {}  # name -> (directory, grid-to-network pixel rows and columns, scale to full-resolution disparity, threshold)
    for spec in args.nn:
        d, _, c = spec.partition(":")
        meta = json.load(open(os.path.join(d, "meta.json")))
        w, h, fw, fh = meta["width"], meta["height"], meta["full_width"], meta["full_height"]
        u = np.clip(np.rint((xs * s + 0.5) * w / fw - 0.5), 0, w - 1).astype(int)
        v = np.clip(np.rint((ys * s + 0.5) * h / fh - 0.5), 0, h - 1).astype(int)
        name = meta["name"] + (f", confidence >= {c}" if c else "")
        nets[name] = (d, (v, u), (h, w), fw / w, float(c) if c else None)
    first_nn = next(iter(nets), None)

    if args.raw:
        K_raw, d_raw, _ = camchain(os.path.join(args.src, "raw/camchain.yaml"))
        R = rectification(args.src, stems, K, K_raw, d_raw)
        # Where each rectified grid pixel's ray lands on the raw grid (same centre: same range).
        uv, _ = cv2.projectPoints(rays(K, xs * s, ys * s).reshape(-1, 3) @ R, np.zeros(3), np.zeros(3), K_raw, d_raw)
        uv = uv.reshape(H, W, 2) / s
        RX, RY = np.round(uv[..., 0]).astype(int), np.round(uv[..., 1]).astype(int)
        inside = (RX >= 0) & (RY >= 0) & (RX < W) & (RY < H)

    others = list(dis) + list(sgbm) + list(nets)
    scores = {"ours": Score(), **{name: Score() for name in others}}
    if args.on_ours:
        scores.update({f"{name}, where ours has depth": Score() for name in others})
    if os.path.exists(os.path.join(args.ours, stems[0] + ".src")):
        scores["ours, finest level only"] = Score()
        scores["ours, coarse fallback only"] = Score()
        scores["ours, semi-global inferred only"] = Score()
    if args.raw:
        scores["ours, raw images"] = Score()
    if args.png:
        os.makedirs(args.png, exist_ok=True)
    dis_ms = {name: [] for name in dis}
    sgbm_ms = {name: [] for name in sgbm}
    resize_ms = []
    for k, stem in enumerate(stems):
        z = cv2.imread(os.path.join(args.src, "rect/depth", stem + ".png"), -1)[::s, ::s][:H, :W] / 1000.0
        sdk = np.where(z > 0, np.log(np.maximum(z, 1e-3) * n_grid), np.nan).astype(np.float32)
        ours = np.fromfile(os.path.join(args.ours, stem + ".logr"), "<f4").reshape(H, W)
        left = cv2.imread(os.path.join(args.src, "rect/left", stem + ".png"), 0)
        right = cv2.imread(os.path.join(args.src, "rect/right", stem + ".png"), 0)
        est = {"ours": ours}
        t0 = time.perf_counter()
        a = cv2.resize(left, (left.shape[1] // sd, left.shape[0] // sd), interpolation=cv2.INTER_AREA)
        b = cv2.resize(right, (right.shape[1] // sd, right.shape[0] // sd), interpolation=cv2.INTER_AREA)
        resize_ms.append((time.perf_counter() - t0) * 1e3)
        for name, d in dis.items():
            t0 = time.perf_counter()
            flow = d.calc(a, b, None)
            dis_ms[name].append((time.perf_counter() - t0) * 1e3)
            disp = -flow[::step, ::step][:H, :W, 0] * sd  # full-resolution pixels
            with np.errstate(all="ignore"):
                est[name] = np.where(disp > 0, np.log(K[0, 0] * baseline / disp * n_dis), np.nan).astype(np.float32)
            est[name][(est[name] < np.log(0.25)) | (est[name] > np.log(30))] = np.nan
        small = {1: (left, right)}  # images downsampled by SGBM's factors
        for name, (sc, m, nd) in sgbm.items():
            if sc not in small:
                small[sc] = tuple(cv2.resize(im, (im.shape[1] // sc, im.shape[0] // sc), interpolation=cv2.INTER_AREA)
                                  for im in (left, right))
            a, b = (cv2.copyMakeBorder(im, 0, 0, nd, 0, cv2.BORDER_CONSTANT) for im in small[sc])
            t0 = time.perf_counter()
            raw_disp = m.compute(a, b)
            sgbm_ms[name].append((time.perf_counter() - t0) * 1e3)
            disp = raw_disp[:, nd:].astype(np.float32) / 16
            disp[disp > np.arange(disp.shape[1])] = 0  # matched left of the right image
            disp = disp[:: s // sc, :: s // sc][:H, :W] * sc
            n_sc = 1 / rays(K, xs * s + 0.5 * (sc - 1), ys * s + 0.5 * (sc - 1))[..., 2]
            with np.errstate(all="ignore"):
                est[name] = np.where(disp > 0, np.log(K[0, 0] * baseline / disp * n_sc), np.nan).astype(np.float32)
            est[name][(est[name] < np.log(0.25)) | (est[name] > np.log(30))] = np.nan
        for name, (d, (v, u), shape, scale, thr) in nets.items():
            disp = np.fromfile(os.path.join(d, stem + ".disp"), "<f2").reshape(shape).astype(np.float32)
            if thr is not None:
                disp[np.fromfile(os.path.join(d, stem + ".conf"), "<f2").reshape(shape) < thr] = 0
            disp = disp[v, u] * scale
            with np.errstate(all="ignore"):
                est[name] = np.where(disp > 0, np.log(K[0, 0] * baseline / disp * n_grid), np.nan).astype(np.float32)
            est[name][(est[name] < np.log(0.25)) | (est[name] > np.log(30))] = np.nan
        d_logr = est[first_dis]
        if args.on_ours:
            for name in others:
                est[f"{name}, where ours has depth"] = np.where(np.isfinite(ours), est[name], np.nan)
        src_path = os.path.join(args.ours, stem + ".src")
        if os.path.exists(src_path):  # source level per pixel: split measured / coarse fallback
            src = np.fromfile(src_path, np.uint8).reshape(H, W)
            for name, sel in (("ours, finest level only", src == frames[0]["level"]),
                              ("ours, coarse fallback only", (src < 254) & (src != frames[0]["level"])),
                              ("ours, semi-global inferred only", src == 254)):
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
                    np.hstack([label(colour(d_logr), f"OpenCV {first_dis} ({pct(d_logr)})", True),
                               label(err_colour(ours - sdk), "ours vs SDK", True)])]
            if args.raw:
                rows.append(np.hstack([label(colour(raw), f"ours on raw images ({pct(raw)})"),
                                       label(err_colour(d_logr - sdk), "DIS vs SDK", True)]))
            if first_sgbm:
                g = est[first_sgbm]
                rows.append(np.hstack([label(colour(g), f"OpenCV {first_sgbm} ({pct(g)})", True),
                                       label(err_colour(g - sdk), "SGBM vs SDK", True)]))
            if first_nn:
                g = est[first_nn]
                rows.append(np.hstack([label(colour(g), f"{first_nn} ({pct(g)})", True),
                                       label(err_colour(g - sdk), "network vs SDK", True)]))
            legend = np.full((30, 2 * W, 3), 255, np.uint8)
            cv2.putText(legend, "range 0.3 m red .. 6 m blue, black = none; error red = nearer than SDK, blue = "
                        "farther, full at 20 %; line: obstacle band", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(args.png, stem + ".png"), np.vstack(rows + [legend]))

    ours_ms = [f["ms"] for f in frames if "ms" in f]
    print(f"{len(stems)} frames, {W}x{H} grid; ms per pair, median: ours {np.median(ours_ms):.1f} (where it ran); "
          + ", ".join(f"{name} {np.median(t):.2f}" for name, t in {**dis_ms, **sgbm_ms}.items())
          + f" plus {np.median(resize_ms):.2f} to downsample (this machine, {cv2.getNumThreads()} OpenCV threads)")
    for name, sc in scores.items():
        report(name, sc)


if __name__ == "__main__":
    main()
