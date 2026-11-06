#!/usr/bin/env python3
"""Record ZED 2i stereo with the SDK's NEURAL_LIGHT depth, for evaluating dense stereo.

Two steps, so walking with the camera never waits on the disk:

  record   python3 zed_record.py record walk.svo2 [--seconds 120] [--res HD720] [--fps 30]
           An SVO2 file, lossless: the raw frames, their timestamps and the factory
           calibration. No depth is computed while recording.

  extract  python3 zed_record.py extract walk.svo2 OUT [--every 3] [--depth NEURAL_LIGHT]
           Replays the SVO with the SDK's depth and writes every N-th frame (file name =
           timestamp in ns):
             OUT/raw/{left,right}/<ns>.png   unrectified grey
             OUT/raw/camchain.yaml            factory calibration, OpenCV rational_polynomial
             OUT/rect/{left,right}/<ns>.png  rectified grey
             OUT/rect/camchain.yaml           the rectified pinhole pair
             OUT/rect/depth/<ns>.png         SDK depth, uint16 millimetres along the rectified
                                              left camera's optical axis (Z); 0 = none
             OUT/rect/confidence/<ns>.png    SDK confidence, uint8, 1 (best) .. 100
             OUT/frames.csv                   stem, depth retrieval ms

OUT/raw and OUT/rect are what dense_stereo_eval reads:
    cargo run --release --example dense_stereo_eval -- OUT/rect --out results

The camera must be free (no ROS node running). The Orin Nano has no hardware video
encoder, so the SVO uses the CPU's LOSSLESS mode; if `record` reports dropped frames,
lower --fps.
"""
import argparse
import os
import time

import cv2
import numpy as np
import pyzed.sl as sl
import yaml

RES = {"HD720": sl.RESOLUTION.HD720, "VGA": sl.RESOLUTION.VGA, "HD1080": sl.RESOLUTION.HD1080}


def record(args):
    cam = sl.Camera()
    init = sl.InitParameters(camera_resolution=RES[args.res], camera_fps=args.fps,
                             depth_mode=sl.DEPTH_MODE.NONE)
    init.camera_disable_self_calib = True  # as zed_slam's vio_feed.yaml: factory calibration only
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        raise SystemExit("Cannot open the camera (is a ROS node still running?)")
    rec = sl.RecordingParameters(args.svo, sl.SVO_COMPRESSION_MODE.LOSSLESS)
    err = cam.enable_recording(rec)
    if err != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"Cannot record: {err}")
    period_ns = 1e9 / args.fps
    frames, dropped, last = 0, 0, None
    t_end = time.monotonic() + args.seconds
    print(f"recording {args.svo} for {args.seconds} s, Ctrl-C to stop early")
    try:
        while time.monotonic() < t_end:
            if cam.grab() != sl.ERROR_CODE.SUCCESS:
                continue
            ts = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            if last is not None:
                dropped += max(0, round((ts - last) / period_ns) - 1)
            last = ts
            frames += 1
    except KeyboardInterrupt:
        pass
    cam.disable_recording()
    cam.close()
    print(f"{frames} frames, {dropped} dropped (from image timestamps)")


def cam_entry(p, model, dist):
    return {"camera_model": "pinhole", "intrinsics": [float(p.fx), float(p.fy), float(p.cx), float(p.cy)],
            "distortion_model": model, "distortion_coeffs": [float(v) for v in dist],
            "resolution": [p.image_size.width, p.image_size.height]}


def cam1_from_cam0(T):
    """Kalibr's T_cn_cnm1 (p_cam1 = R p_cam0 + t) from the SDK's stereo_transform.

    The SDK's rotation already maps left-camera points into the right camera (it is
    zed-open-capture's Rodrigues(RX, CV, RZ)); its translation is (Baseline, TY, TZ),
    while the right camera's is (-Baseline, TY, TZ). Inverting the whole transform (as
    zed_slam's export_calib.py does) transposes the rotation: on raw ZED 2i images that
    leaves matches 4.6 px off their epipolar lines, this 0.09 px.
    """
    T = np.array(T, dtype=float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3]
    out[:3, 3] = [-T[0, 3], T[1, 3], T[2, 3]]
    return out


def write_camchains(cam, out, serial):
    cfg = cam.get_camera_information().camera_configuration
    rect, raw = cfg.calibration_parameters, cfg.calibration_parameters_raw
    chains = {
        "rect": {"cam0": cam_entry(rect.left_cam, "none", [0.0] * 4),
                 "cam1": cam_entry(rect.right_cam, "none", [0.0] * 4)},
        "raw": {"cam0": cam_entry(raw.left_cam, "rational_polynomial", raw.left_cam.disto[:8]),
                "cam1": cam_entry(raw.right_cam, "rational_polynomial", raw.right_cam.disto[:8])},
    }
    chains["rect"]["cam1"]["T_cn_cnm1"] = np.round(cam1_from_cam0(rect.stereo_transform.m), 9).tolist()
    chains["raw"]["cam1"]["T_cn_cnm1"] = np.round(cam1_from_cam0(raw.stereo_transform.m), 9).tolist()
    for name, chain in chains.items():
        with open(os.path.join(out, name, "camchain.yaml"), "w") as f:
            f.write(f"# ZED S/N {serial}, factory calibration from ZED SDK {sl.Camera.get_sdk_version()}\n")
            yaml.safe_dump(chain, f, default_flow_style=None, sort_keys=False)


def extract(args):
    for sub in ("raw/left", "raw/right", "rect/left", "rect/right", "rect/depth", "rect/confidence"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    cam = sl.Camera()
    init = sl.InitParameters(depth_mode=getattr(sl.DEPTH_MODE, args.depth),
                             coordinate_units=sl.UNIT.METER, coordinate_system=sl.COORDINATE_SYSTEM.IMAGE)
    init.set_from_svo_file(args.svo)
    init.svo_real_time_mode = False
    init.camera_disable_self_calib = True
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"Cannot open {args.svo}")
    write_camchains(cam, args.out, cam.get_camera_information().serial_number)
    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    views = [("raw/left", sl.VIEW.LEFT_UNRECTIFIED_GRAY), ("raw/right", sl.VIEW.RIGHT_UNRECTIFIED_GRAY),
             ("rect/left", sl.VIEW.LEFT_GRAY), ("rect/right", sl.VIEW.RIGHT_GRAY)]
    n, saved = 0, 0
    with open(os.path.join(args.out, "frames.csv"), "w") as csv:
        csv.write(f"# depth_mode {args.depth}, confidence_threshold {runtime.confidence_threshold}\n")
        csv.write("stem,depth_ms\n")
        while True:
            t0 = time.perf_counter()
            err = cam.grab(runtime)
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if err != sl.ERROR_CODE.SUCCESS:
                continue
            ms = (time.perf_counter() - t0) * 1e3
            n += 1
            if (n - 1) % args.every:
                continue
            stem = str(cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            for sub, view in views:
                cam.retrieve_image(mat, view)
                cv2.imwrite(os.path.join(args.out, sub, stem + ".png"), mat.get_data())
            cam.retrieve_measure(mat, sl.MEASURE.DEPTH_U16_MM)
            cv2.imwrite(os.path.join(args.out, "rect/depth", stem + ".png"), mat.get_data())
            cam.retrieve_measure(mat, sl.MEASURE.CONFIDENCE)
            conf = np.nan_to_num(mat.get_data(), nan=100.0)
            cv2.imwrite(os.path.join(args.out, "rect/confidence", stem + ".png"),
                        np.clip(conf, 0, 100).astype(np.uint8))
            csv.write(f"{stem},{ms:.1f}\n")
            saved += 1
    cam.close()
    print(f"{n} frames in the SVO, {saved} written to {args.out}")


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("svo")
    r.add_argument("--seconds", type=float, default=120.0)
    r.add_argument("--res", default="HD720", choices=sorted(RES))
    r.add_argument("--fps", type=int, default=30)
    e = sub.add_parser("extract")
    e.add_argument("svo")
    e.add_argument("out")
    e.add_argument("--every", type=int, default=3)
    e.add_argument("--depth", default="NEURAL_LIGHT")
    args = ap.parse_args()
    record(args) if args.cmd == "record" else extract(args)


if __name__ == "__main__":
    main()
