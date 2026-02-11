#!/usr/bin/env python3
"""
capture_oak_rgbd.py

Capture synchronized-ish RGB + aligned depth from an OAK-D Lite (DepthAI 2.x),
and write into a run folder:

  --out/
    color/color_000000.jpg
    depth/depth_000000.png   (uint16, mm)
    timestamps.csv           (idx, t_wall_s, t_rgb_dev_ms, t_depth_dev_ms)
    intrinsics.json          (fx, fy, cx, cy, width, height)

Press 'q' in the preview window to stop.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import depthai as dai


CAP_W, CAP_H = 640, 400


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture OAK RGB + aligned depth -> folder")
    p.add_argument("--out", required=True, help="Output directory, e.g. $RUN_DIR/raw/oak_rgbd")
    p.add_argument("--fps", type=int, default=15, help="FPS (default: 15)")
    p.add_argument("--no-preview", action="store_true", help="Disable OpenCV preview windows")
    return p.parse_args()


def save_intrinsics(device: dai.Device, out_dir: Path, w: int, h: int) -> None:
    fx = fy = 600.0
    cx, cy = w / 2.0, h / 2.0
    source = "approximate"
    try:
        calib = device.readCalibration()
        M = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, w, h)
        fx, fy, cx, cy = float(M[0][0]), float(M[1][1]), float(M[0][2]), float(M[1][2])
        source = "oak_calibration"
    except Exception as e:
        print(f"WARNING: Could not read calibration intrinsics ({e}); using approximate intrinsics.")

    data = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": w, "height": h, "source": source}
    (out_dir / "intrinsics.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Intrinsics saved -> {out_dir / 'intrinsics.json'} ({source})")


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    (out / "color").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    pipeline = dai.Pipeline()

    # --- RGB ---
    color = pipeline.create(dai.node.ColorCamera)
    color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    color.setInterleaved(False)
    color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    if hasattr(color, "setVideoSize"):
        color.setVideoSize(CAP_W, CAP_H)
    if hasattr(color, "setFps"):
        color.setFps(args.fps)

    # --- Stereo depth (aligned to RGB) ---
    monoL = pipeline.create(dai.node.MonoCamera)
    monoR = pipeline.create(dai.node.MonoCamera)
    monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)     # align depth to RGB
    stereo.setOutputSize(CAP_W, CAP_H)

    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    # --- Outputs (THIS is what makes queues exist) ---
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    color.video.link(xoutRgb.input)

    xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutDepth.setStreamName("depth")
    stereo.depth.link(xoutDepth.input)

    print(f"Starting capture -> {out} (press q to quit)")

    with dai.Device(pipeline) as device:
        save_intrinsics(device, out, CAP_W, CAP_H)

        qRgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        qDepth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        with open(out / "timestamps.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "t_wall_s", "t_rgb_dev_ms", "t_depth_dev_ms"])

            idx = 0
            while True:
                inRgb = qRgb.tryGet()
                inDepth = qDepth.tryGet()
                if inRgb is None or inDepth is None:
                    time.sleep(0.001)
                    continue

                rgb = inRgb.getCvFrame()     # BGR uint8
                depth = inDepth.getFrame()   # uint16 depth (mm)

                cv2.imwrite(str(out / "color" / f"color_{idx:06d}.jpg"), rgb)
                cv2.imwrite(str(out / "depth" / f"depth_{idx:06d}.png"), depth)

                t_wall = time.time()
                t_rgb = int(inRgb.getTimestampDevice().total_seconds() * 1000.0)
                t_dep = int(inDepth.getTimestampDevice().total_seconds() * 1000.0)
                w.writerow([idx, t_wall, t_rgb, t_dep])

                if not args.no_preview:
                    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imshow("rgb", rgb)
                    cv2.imshow("depth_vis", depth_vis)

                idx += 1
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    print(f"Done. Saved {idx} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
