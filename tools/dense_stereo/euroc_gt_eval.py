#!/usr/bin/env python3
"""Score dense stereo depth on EuRoC V1/V2 against the room's laser scan, next to OpenCV DIS.

    cargo run --release --example dense_stereo_eval -- V1_01_easy --frames 0:2912:100 --out out \
        > out/frames.jsonl
    python3 euroc_gt_eval.py V1_01_easy out [--dis-preset fast|medium|ultrafast] [--dis-scale 1.0]
        [--level 1] [--bottom 8]

Truth: pointcloud0 (Leica scan, world frame) seen from cam0's ground-truth pose, z-buffered
at the depth map's resolution; returns that leak through gaps between a near surface's
points are dropped. DIS: L->R flow on the raw images, triangulated (both rays through the
radtan model, undistorted by Newton). Prints coverage, log-range errors overall and by
range, and the nearest range per image column (the obstacle bins' quantity) too near or
too far by more than 10 %, over all but the bottom --bottom rows.

CAVEAT: the ground-truth poses and the scan disagree by 4-6 degrees per frame on V1_01
(ICP of the stereo depth onto the scan; no time offset or single global transform fixes
it), and the radtan calibration is off near the image corners. Two stereo methods agree
with each other to 2-4 % but each disagrees with this truth by 6-18 %: use it to compare
methods on the same frames, not for absolute accuracy.
"""
import json
import os
import sys
import time

import cv2
import numpy as np


def read_yaml_matrix(path, key):
    txt = open(path).read()
    i = txt.index(key)
    j = txt.index("data:", i)
    k = txt.index("]", j)
    return np.array([float(v) for v in txt[txt.index("[", j) + 1:k].replace("\n", " ").split(",")])


def read_list(path, key):
    for line in open(path):
        if line.strip().startswith(key):
            return np.array([float(v) for v in line.split("[")[1].split("]")[0].split(",")])
    raise KeyError(key)


def camera(path):
    T = read_yaml_matrix(path, "T_BS").reshape(4, 4)
    fx, fy, cx, cy = read_list(path, "intrinsics:")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    dist = read_list(path, "distortion_coefficients:")
    return T, K, dist


def _undistort(pts, K, dist):
    """Normalised coordinates by Newton (cv2.undistortPoints stops after 5 fixed-point
    iterations: 0.3 px off near EuRoC's corners)."""
    pts = pts.reshape(-1, 2).astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, p1, p2 = dist[:4]
    tx, ty = (pts[:, 0] - cx) / fx, (pts[:, 1] - cy) / fy
    x, y = tx.copy(), ty.copy()
    for _ in range(20):  # Newton on the radial-tangential model
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2
        dr = k1 + 2 * k2 * r2
        ex = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) - tx
        ey = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y - ty
        a = rad + 2 * x * x * dr + 2 * p1 * y + 6 * p2 * x
        b = 2 * x * y * dr + 2 * p1 * x + 2 * p2 * y
        d = rad + 2 * y * y * dr + 6 * p1 * y + 2 * p2 * x
        det = a * d - b * b
        x -= (d * ex - b * ey) / det
        y -= (a * ey - b * ex) / det
    return np.stack([x, y], -1).reshape(-1, 1, 2)


def quat_to_R(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_ply(path):
    cache = path + ".npy"
    if os.path.exists(cache):
        return np.load(cache)
    with open(path) as f:
        while f.readline().strip() != "end_header":
            pass
        pts = np.loadtxt(f, usecols=(0, 1, 2), dtype=np.float32)
    np.save(cache, pts)
    return pts


def gt_logrange(pts_w, T_WC, K, dist, w, h, level):
    """Nearest-surface log-range per depth-map pixel; NaN where the scan has nothing."""
    R, t = T_WC[:3, :3], T_WC[:3, 3]
    pc = (pts_w - t) @ R  # world -> camera: R^T (p - t)
    pc = pc[pc[:, 2] > 0.1]
    uv, _ = cv2.projectPoints(pc.astype(np.float64), np.zeros(3), np.zeros(3), K, dist)
    uv = uv.reshape(-1, 2)
    s = 2 ** level
    x = np.round(uv[:, 0] / s).astype(int)
    y = np.round(uv[:, 1] / s).astype(int)
    r = np.linalg.norm(pc, axis=1)
    zb = np.full((h, w), np.inf, np.float32)
    # Splat 3x3 so sparse foreground still hides the background behind it.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            xx, yy = x + dx, y + dy
            ok = (xx >= 0) & (yy >= 0) & (xx < w) & (yy < h)
            np.minimum.at(zb, (yy[ok], xx[ok]), r[ok])
    # Count of points that land exactly; a pixel needs some support.
    cnt = np.zeros((h, w), np.int32)
    ok = (x >= 0) & (y >= 0) & (x < w) & (y < h)
    np.add.at(cnt, (y[ok], x[ok]), 1)
    # Background returns leak through gaps between a near surface's points: keep a pixel only
    # if it is within 5 % of the nearest return around it.
    near = cv2.erode(np.where(np.isfinite(zb), zb, 1e6).astype(np.float32), np.ones((7, 7), np.uint8))
    keep = np.isfinite(zb) & (cnt > 0) & (zb <= near * 1.05)
    out = np.where(keep, np.log(zb), np.nan).astype(np.float32)
    return out


def dis_logrange(left, right, K0, d0, K1, d1, R10, t10, w, h, level, preset, scale):
    presets = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
               "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}
    dis = cv2.DISOpticalFlow_create(presets[preset])
    dis.setUseSpatialPropagation(True)
    a, b = left, right
    if scale < 1.0:
        size = (int(round(left.shape[1] * scale)), int(round(left.shape[0] * scale)))
        a = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
        b = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
    t0 = time.perf_counter()
    flow = dis.calc(a, b, None)
    ms = (time.perf_counter() - t0) * 1e3
    s = 2 ** level
    ys, xs = np.mgrid[0:h, 0:w]
    u0 = (xs * s).astype(np.float64)
    v0 = (ys * s).astype(np.float64)
    fx = cv2.resize(flow[..., 0], (left.shape[1], left.shape[0])) / scale
    fy = cv2.resize(flow[..., 1], (left.shape[1], left.shape[0])) / scale
    u1 = u0 + fx[ys * s, xs * s]
    v1 = v0 + fy[ys * s, xs * s]
    b0 = _undistort(np.stack([u0, v0], -1).reshape(-1, 1, 2), K0, d0).reshape(-1, 2)
    b1 = _undistort(np.stack([u1, v1], -1).reshape(-1, 1, 2), K1, d1).reshape(-1, 2)
    b0 = np.c_[b0, np.ones(len(b0))]
    b0 /= np.linalg.norm(b0, axis=1, keepdims=True)
    b1 = np.c_[b1, np.ones(len(b1))]
    b1 = b1 @ R10  # R^T b1: cam1 ray in cam0
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    c1 = -R10.T @ t10
    # Midpoint: r0 b0 - r1 b1 = c1.
    bb = np.sum(b0 * b1, 1)
    d0c = b0 @ c1
    d1c = b1 @ c1
    den = 1 - bb * bb
    r0 = (d0c - bb * d1c) / np.maximum(den, 1e-12)
    out = np.where((den > 1e-8) & (r0 > 0.05), np.log(np.maximum(r0, 1e-6)), np.nan)
    return out.reshape(h, w).astype(np.float32), ms


def score(est, gt, bottom_rows=0):
    m = np.isfinite(gt)
    has = m & np.isfinite(est)
    e = np.abs(est[has] - gt[has])
    res = {"gt_px": int(m.sum()), "coverage": float(has.sum() / max(m.sum(), 1)),
           "median": float(np.median(e)) if len(e) else np.nan,
           "bad10": float(np.mean(e > np.log(1.1))) if len(e) else np.nan,
           "bad25": float(np.mean(e > np.log(1.25))) if len(e) else np.nan}
    g = np.exp(gt[has])
    for lo, hi in [(0, 1.5), (1.5, 2.5), (2.5, 4), (4, 99)]:
        sel = (g >= lo) & (g < hi)
        res[f"med_{lo}-{hi}"] = float(np.median(e[sel])) if sel.sum() > 50 else np.nan
        res[f"n_{lo}-{hi}"] = int(sel.sum())
    # Nearest range per column, where the scan has at least 20 % of the column.
    col_err = []
    col_miss = 0
    rows = gt.shape[0] - bottom_rows
    for x in range(gt.shape[1]):
        gcol = gt[:rows, x]
        if np.isfinite(gcol).mean() < 0.2:
            continue
        ecol = est[:rows, x]
        if not np.isfinite(ecol).any():
            col_miss += 1
            continue
        col_err.append(np.nanmin(ecol) - np.nanmin(gcol))
    col_err = np.array(col_err)
    res["col_n"] = int(len(col_err) + col_miss)
    res["col_missing"] = col_miss
    res["col_med_abs"] = float(np.median(np.abs(col_err))) if len(col_err) else np.nan
    # Nearer than the truth by > 10 % (phantom obstacle) / farther by > 10 % (missed obstacle).
    res["col_too_near"] = float(np.mean(col_err < -np.log(1.1))) if len(col_err) else np.nan
    res["col_too_far"] = float(np.mean(col_err > np.log(1.1))) if len(col_err) else np.nan
    return res


def main():
    args = sys.argv[1:]
    def opt(name, default):
        if name in args:
            i = args.index(name)
            v = args[i + 1]
            del args[i:i + 2]
            return v
        return default
    preset = opt("--dis-preset", "fast")
    scale = float(opt("--dis-scale", "1.0"))
    level = int(opt("--level", "1"))
    # Nearest-per-column ignores the bottom band no patch fits in (8 px of the level).
    bottom = int(opt("--bottom", "8"))
    seq, ours_dir = args[0], args[1]
    mav0 = os.path.join(seq, "mav0")
    T_B0, K0, d0 = camera(os.path.join(mav0, "cam0/sensor.yaml"))
    T_B1, K1, d1 = camera(os.path.join(mav0, "cam1/sensor.yaml"))
    T_10 = np.linalg.inv(T_B1) @ T_B0
    R10, t10 = T_10[:3, :3], T_10[:3, 3]
    gtp = np.loadtxt(os.path.join(mav0, "state_groundtruth_estimate0/data.csv"), delimiter=",", comments="#")
    pts = load_ply(os.path.join(mav0, "pointcloud0/data.ply"))

    frames = [json.loads(l) for l in open(os.path.join(ours_dir, "frames.jsonl"))]
    agg = {"ours": [], "dis": []}
    dis_ms = []
    for f in frames:
        stem = f["stem"]
        w, h = f["width"], f["height"]
        ts = int(stem)
        i = np.argmin(np.abs(gtp[:, 0] - ts))
        if abs(gtp[i, 0] - ts) > 5e6:
            continue
        T_WB = np.eye(4)
        T_WB[:3, :3] = quat_to_R(*gtp[i, 4:8])
        T_WB[:3, 3] = gtp[i, 1:4]
        gt = gt_logrange(pts, T_WB @ T_B0, K0, d0, w, h, level)
        ours = np.fromfile(os.path.join(ours_dir, stem + ".logr"), dtype="<f4").reshape(h, w)
        left = cv2.imread(os.path.join(mav0, "cam0/data", stem + ".png"), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(os.path.join(mav0, "cam1/data", stem + ".png"), cv2.IMREAD_GRAYSCALE)
        dis, ms = dis_logrange(left, right, K0, d0, K1, d1, R10, t10, w, h, level, preset, scale)
        dis_ms.append(ms)
        agg["ours"].append(score(ours, gt, bottom))
        agg["dis"].append(score(dis, gt, bottom))
        np.save(os.path.join(ours_dir, stem + ".gt.npy"), gt)
        np.save(os.path.join(ours_dir, stem + ".dis.npy"), dis)

    keys = ["coverage", "median", "bad10", "bad25", "med_0-1.5", "med_1.5-2.5", "med_2.5-4", "med_4-99",
            "col_med_abs", "col_too_near", "col_too_far"]
    print(f"{len(agg['ours'])} frames; DIS {preset} at scale {scale}: {np.median(dis_ms):.1f} ms median (this machine)")
    print(f"{'':10s}" + "".join(f"{k:>13s}" for k in keys))
    for name, rows in agg.items():
        vals = [np.nanmean([r[k] for r in rows]) for k in keys]
        print(f"{name:10s}" + "".join(f"{v:13.4f}" for v in vals))
    miss = {n: sum(r["col_missing"] for r in rows) for n, rows in agg.items()}
    ncol = sum(r["col_n"] for r in agg["ours"])
    print(f"columns with scan but no estimate at all: ours {miss['ours']} / {ncol}, dis {miss['dis']} / {ncol}")


if __name__ == "__main__":
    main()
