"""Stereo pairs with a checkerboard pose, from a ZED board bag (rect grey images, camera_info).

usage: board_extract.py DB3 OUT EVERY SQUARE_M
Every EVERY-th left/right pair (matched by stamp) in which Rudolf-V's detector (echo_li) finds the
whole 9x6 board in the left image: OUT/rect/{left,right}/<stamp ns>.png, OUT/rect/camchain.yaml
(Kalibr, as zed_record.py writes it) and OUT/board.json (per frame: corners, PnP pose with
SQUARE_M squares, reprojection RMS). Run with ROS sourced and ~/mrnn_test/venv_echoli's python."""
import json
import os
import sys

import cv2
import numpy as np
import rosbag2_py
from echo_li import CheckerboardDetector
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

db3, out, every, square = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
COLS, ROWS = 9, 6
Z = "/zed/zed_node/"
TOPICS = {Z + "left/gray/rect/image": "left", Z + "right/gray/rect/image": "right",
          Z + "left/gray/rect/camera_info": "left_info", Z + "right/gray/rect/camera_info": "right_info"}
OBJ = np.zeros((COLS * ROWS, 3), np.float32)
OBJ[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * square  # row-major, columns along +x

for eye in ("left", "right"):
    os.makedirs(os.path.join(out, "rect", eye), exist_ok=True)
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=db3, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
r.set_filter(rosbag2_py.StorageFilter(topics=list(TOPICS)))
types = {t.name: get_message(t.type) for t in r.get_all_topics_and_types()}
det = CheckerboardDetector(COLS, ROWS)
info, pending, frames = {}, {}, []
pairs = 0
while r.has_next():
    topic, data, _ = r.read_next()
    kind = TOPICS[topic]
    m = deserialize_message(data, types[topic])
    if kind.endswith("info"):
        info.setdefault(kind[:-5], m)
        continue
    stamp = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
    pending.setdefault(stamp, {})[kind] = m
    if len(pending[stamp]) < 2:
        continue
    pair = pending.pop(stamp)
    for s in [s for s in pending if s < stamp]:  # the other eye's frame was lost
        del pending[s]
    pairs += 1
    if pairs % every or "left" not in info:
        continue
    imgs = {e: np.frombuffer(pair[e].data, np.uint8).reshape(pair[e].height, pair[e].step)[:, :pair[e].width]
            for e in ("left", "right")}
    corners = det.detect(np.ascontiguousarray(imgs["left"]))
    if corners is None:
        continue
    K = np.array(info["left"].k).reshape(3, 3)
    ok, rvec, tvec = cv2.solvePnP(OBJ, corners.astype(np.float64), K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        continue
    proj, _ = cv2.projectPoints(OBJ, rvec, tvec, K, None)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, 1))))
    for e in ("left", "right"):
        cv2.imwrite(os.path.join(out, "rect", e, f"{stamp}.png"), imgs[e])
    frames.append({"stem": str(stamp), "rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist(),
                   "rms_px": round(rms, 4), "corners": np.round(corners, 3).tolist()})

K = np.array(info["left"].k).reshape(3, 3)
P1 = np.array(info["right"].p).reshape(3, 4)
baseline = -P1[0, 3] / P1[0, 0]
cam = {"camera_model": "pinhole", "intrinsics": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
       "distortion_model": "none", "distortion_coeffs": [0.0] * 4,
       "resolution": [int(info["left"].width), int(info["left"].height)]}
with open(os.path.join(out, "rect", "camchain.yaml"), "w") as f:
    f.write("# from the bag's rect camera_info (left K, baseline from the right P)\ncam0:\n")
    for k, v in cam.items():
        f.write(f"  {k}: {json.dumps(v) if not isinstance(v, str) else v}\n")
    f.write("cam1:\n")
    for k, v in cam.items():
        f.write(f"  {k}: {json.dumps(v) if not isinstance(v, str) else v}\n")
    f.write(f"  T_cn_cnm1:\n  - [1.0, 0.0, 0.0, {-baseline:.9f}]\n  - [0.0, 1.0, 0.0, 0.0]\n"
            "  - [0.0, 0.0, 1.0, 0.0]\n  - [0.0, 0.0, 0.0, 1.0]\n")
json.dump({"square_m": square, "pattern": [COLS, ROWS], "every": every, "K": K.tolist(), "baseline_m": baseline,
           "frames": frames}, open(os.path.join(out, "board.json"), "w"))
print(f"{pairs} pairs, kept every {every}th: board found in {len(frames)}; baseline {baseline:.6f} m")
