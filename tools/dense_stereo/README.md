# Dense stereo evaluation tools

Python helpers around `examples/dense_stereo_eval.rs` (issue #5). Each script's docstring has its
full usage.

| Script | Runs on | Does |
|---|---|---|
| `zed_record.py` | the Orin with the ZED | `record`: lossless SVO2 of a walk; `extract`: raw and rectified pairs plus the SDK's NEURAL_LIGHT depth, as folders `dense_stereo_eval` reads |
| `zed_eval.py` | anywhere | a `zed_record.py extract` folder against the SDK's depth, next to OpenCV DIS as the robot pipeline runs it (or other DIS presets and scales, `--dis`; timed on `calc()` alone, `--threads`): errors by range, the obstacle band (nearest per column, risk bins, blind bins), measured vs coarse-fallback pixels, OpenCV SGBM (`--sgbm`), network disparity (`--nn`), and the others on the pixels ours has depth (`--on-ours`) |
| `nn_infer.py` | the Orin | a stereo network's TensorRT engine (Light ESS, or LAS2 at any input size) over a recording's rectified pairs: its disparity (and confidence) per frame, which `zed_eval.py --nn` scores |
| `nn_time.py` | the Orin | an engine timed as a 30 fps pipeline calls it (one inference per frame period, own CUDA stream, optionally a CUDA graph) and back to back |
| `board_extract.py` | the Orin | stereo pairs with the 9x6 board's pose (Rudolf-V's detector through echo_li, PnP) from a ZED board bag |
| `board_gt.py` | anywhere | ground truth on the board's plane from those poses, written as `zed_eval.py` reads the SDK's depth (the whiteboard's white area, or the checker squares) |
| `extract_ros2_bag.py` | anywhere | stereo pairs from the iDog ROS 2 bags (rectified grey) |
| `euroc_gt_eval.py` | anywhere | EuRoC V1/V2 against the Leica scan, next to OpenCV DIS (relative comparisons only, see its caveat) |

ZED raw calibration: use the camchain `zed_record.py extract` writes. Its stereo extrinsic follows
zed-open-capture (R = Rodrigues(RX, CV, RZ), t = (-Baseline, TY, TZ), p_right = R p_left + t).
zed_slam's `tools/export_calib.py` writes the raw rotation transposed, which puts raw pairs about
5 px off their epipolar lines; its rectified camchain is fine.

Networks (2026-11-07): Light ESS from NGC `nvidia/isaac/dnn_stereo_disparity` 4.1.0_onnx (TensorRT
10.3, with its `ess_plugins.so`); LAS2-S/M exported by the LiteAnyStereo repository's
`export_onnx.py` at a fixed input size, max_disp 192. Engines: `trtexec --onnx=M.onnx --fp16
--saveEngine=M.engine` on the Orin (about ten minutes each at 15 W). Check every network's terms
before shipping it: LAS2's weights are MIT, but part of its training used FoundationStereo's
research-only pseudo-labels.
