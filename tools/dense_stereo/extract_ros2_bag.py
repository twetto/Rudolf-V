#!/usr/bin/env python3
"""Extract left/right image pairs from a ROS 2 bag (sqlite3 .db3, sensor_msgs/CompressedImage).

    python3 extract_ros2_bag.py BAG.db3 OUT [--every 150] [--count 40]
        [--left /zed/zed_node/left/gray/rect/image/compressed]
        [--right /zed/zed_node/right/gray/rect/image/compressed]

Writes OUT/left/<stamp ns>.png and OUT/right/<stamp ns>.png for frames whose header stamps
match. Copy the matching Kalibr camchain to OUT/camchain.yaml and dense_stereo_eval reads OUT.
No ROS installation needed: the CDR messages are parsed here.
"""
import argparse
import os
import sqlite3
import struct

import cv2
import numpy as np


def parse_compressed_image(blob):
    """(stamp ns, image bytes) of a CDR sensor_msgs/CompressedImage."""
    o = 4  # encapsulation header
    sec, nsec = struct.unpack_from("<iI", blob, o)
    o += 8
    n = struct.unpack_from("<I", blob, o)[0]  # frame_id
    o = (o + 4 + n + 3) & ~3
    n = struct.unpack_from("<I", blob, o)[0]  # format
    o = (o + 4 + n + 3) & ~3
    n = struct.unpack_from("<I", blob, o)[0]  # data
    o += 4
    return sec * 1_000_000_000 + nsec, blob[o:o + n]


def main():
    ap = argparse.ArgumentParser(description="Extract stereo pairs from a ROS 2 bag.")
    ap.add_argument("bag")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=150)
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--left", default="/zed/zed_node/left/gray/rect/image/compressed")
    ap.add_argument("--right", default="/zed/zed_node/right/gray/rect/image/compressed")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.bag}?mode=ro", uri=True)
    topics = {name: tid for tid, name in con.execute("select id, name from topics")}

    def frames(topic):
        query = "select data from messages where topic_id=? order by timestamp"
        for (data,) in con.execute(query, (topics[topic],)):
            yield parse_compressed_image(data)

    right = dict(frames(args.right))
    for side in ("left", "right"):
        os.makedirs(os.path.join(args.out, side), exist_ok=True)
    k = saved = 0
    for stamp, data in frames(args.left):
        if stamp not in right:
            continue
        k += 1
        if k % args.every:
            continue
        for side, blob in (("left", data), ("right", right[stamp])):
            img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(args.out, side, f"{stamp}.png"), img)
        saved += 1
        if saved >= args.count:
            break
    print(f"saved {saved} pairs to {args.out}")


if __name__ == "__main__":
    main()
