# Dense stereo evaluation tools

Python helpers around `examples/dense_stereo_eval.rs` (issue #5). Each script's docstring has its
full usage.

| Script | Runs on | Does |
|---|---|---|
| `zed_record.py` | the Orin with the ZED | `record`: lossless SVO2 of a walk; `extract`: raw and rectified pairs plus the SDK's NEURAL_LIGHT depth, as folders `dense_stereo_eval` reads |
| `zed_eval.py` | anywhere | a `zed_record.py extract` folder against the SDK's depth, next to OpenCV DIS as the robot pipeline runs it (or other DIS presets and scales, `--dis`; timed on `calc()` alone, `--threads`): errors by range, the obstacle band (nearest per column, risk bins, blind bins), measured vs coarse-fallback pixels, and DIS on the pixels ours has depth (`--on-ours`) |
| `extract_ros2_bag.py` | anywhere | stereo pairs from the iDog ROS 2 bags (rectified grey) |
| `euroc_gt_eval.py` | anywhere | EuRoC V1/V2 against the Leica scan, next to OpenCV DIS (relative comparisons only, see its caveat) |

ZED raw calibration: use the camchain `zed_record.py extract` writes. Its stereo extrinsic follows
zed-open-capture (R = Rodrigues(RX, CV, RZ), t = (-Baseline, TY, TZ), p_right = R p_left + t).
zed_slam's `tools/export_calib.py` writes the raw rotation transposed, which puts raw pairs about
5 px off their epipolar lines; its rectified camchain is fine.
