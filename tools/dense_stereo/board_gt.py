"""Ground-truth depth on the board's plane from each frame's checkerboard pose (board_extract.py),
written as zed_eval.py reads the SDK's: OUT/rect/depth/<stem>.png, uint16 millimetres of Z, 0 where
there is no truth; OUT/rect/{left,right,camchain.yaml} link to SRC's.

usage: board_gt.py SRC OUT REGION
REGION white:   the whiteboard's white area, 2 cm in from its frame (3 cm at the top, where it is
                held), less the checkerboard paper and its tape
       checker: the printed squares
Board coordinates in metres: the first inner corner at the origin, x along the rows (9 corners), y
down the columns. Pixels darker than 0.7 x the region's median grey in that frame are left out (a
hand or arm in front of the board)."""
import json
import os
import sys

import cv2
import numpy as np

src, out, region = sys.argv[1:4]
WHITE = (-0.180, 0.345, -0.120, 0.245)
PAPER = (-0.095, 0.265, -0.070, 0.195)
SQ = 0.0237
CHECKER = (-SQ, 9 * SQ, -SQ, 6 * SQ)
b = json.load(open(os.path.join(src, "board.json")))
K = np.array(b["K"])
Kinv = np.linalg.inv(K)
H, W = 720, 1280
vs, us = np.mgrid[0:H, 0:W]
rays = np.stack([us, vs, np.ones_like(us)], -1).astype(np.float64) @ Kinv.T  # z = 1

os.makedirs(os.path.join(out, "rect", "depth"), exist_ok=True)
for name in ("left", "right", "camchain.yaml"):
    link = os.path.join(out, "rect", name)
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(os.path.join(src, "rect", name)), link)


def poly(R, t, rect):
    x0, x1, y0, y1 = rect
    pts = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], np.float64)
    cam = pts @ R.T + t
    uv = cam @ K.T
    return np.round(uv[:, :2] / uv[:, 2:]).astype(np.int32)


meta = []
for f in b["frames"]:
    R = cv2.Rodrigues(np.array(f["rvec"]))[0]
    t = np.array(f["tvec"])
    mask = np.zeros((H, W), np.uint8)
    if region == "white":
        cv2.fillPoly(mask, [poly(R, t, WHITE)], 1)
        cv2.fillPoly(mask, [poly(R, t, PAPER)], 0)
    else:
        cv2.fillPoly(mask, [poly(R, t, CHECKER)], 1)
    grey = cv2.imread(os.path.join(src, "rect/left", f["stem"] + ".png"), 0)
    m = mask.astype(bool)
    if region == "white" and m.any():
        m &= grey >= 0.7 * np.median(grey[m])
    n = R[:, 2]
    with np.errstate(all="ignore"):
        z = (n @ t) / (rays @ n)
    depth = np.where(m & (z > 0), np.round(z * 1000), 0).astype(np.uint16)
    cv2.imwrite(os.path.join(out, "rect/depth", f["stem"] + ".png"), depth)
    centre = R @ np.array([4 * SQ, 2.5 * SQ, 0]) + t
    tilt = np.degrees(np.arccos(abs(n[2])))
    meta.append({"stem": f["stem"], "z_centre_m": round(float(centre[2]), 4), "tilt_deg": round(float(tilt), 1),
                 "px": int((depth > 0).sum()), "rms_px": f["rms_px"]})
json.dump(meta, open(os.path.join(out, "gt.json"), "w"))
z = np.array([m["z_centre_m"] for m in meta])
print(f"{len(meta)} frames; board centre Z {z.min():.2f}-{z.max():.2f} m (median {np.median(z):.2f}); "
      f"median {int(np.median([m['px'] for m in meta]))} truth px per frame at full resolution")
