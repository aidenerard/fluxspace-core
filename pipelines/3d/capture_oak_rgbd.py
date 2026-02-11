#!/usr/bin/env python3
"""
capture_oak_rgbd.py

Record synchronised RGB + aligned depth frames from a Luxonis OAK-D Lite
(or any OAK-D variant) using the DepthAI SDK.

Outputs (inside --out directory):
  color/color_000000.jpg  ...  colour frames (BGR uint8)
  depth/depth_000000.png  ...  16-bit depth in millimetres
  timestamps.csv               idx, t_wall_s, t_device_ms
  intrinsics.json              pinhole camera intrinsics (fx, fy, cx, cy, width, height)

The output layout is directly consumable by open3d_reconstruct.py and fits
into the standard 3D run folder when --out is set to $RUN_DIR/raw/oak_rgbd.

Matches fluxspace style: argparse, clear prints, sane defaults.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import depthai as dai


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_OUT = "oak_capture"
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 400


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OAK-D Lite RGB+depth capture -> colour frames, 16-bit depth PNGs, timestamps CSV, intrinsics JSON"
    )
    p.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output directory for frames, timestamps and intrinsics. Default: {DEFAULT_OUT}",
    )
    p.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the live OpenCV preview windows (useful for headless / Pi capture)",
    )
    return p.parse_args()


def save_intrinsics(device: dai.Device, out_dir: Path,
                    width: int, height: int) -> None:
    """Read real calibration from the OAK-D and save as intrinsics.json.

    Falls back to approximate values if calibration cannot be read.
    """
    fx = fy = 600.0
    cx = width / 2.0
    cy = height / 2.0
    source = "approximate"

    try:
        calib = device.readCalibration()
        M = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, width, height)
        # M is a 3x3 list-of-lists: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        fx = float(M[0][0])
        fy = float(M[1][1])
        cx = float(M[0][2])
        cy = float(M[1][2])
        source = "oak_calibration"
    except Exception as exc:
        print(f"WARNING: Could not read OAK-D calibration ({exc}); using approximate intrinsics.")

    data = {
        "fx": fx, "fy": fy,
        "cx": cx, "cy": cy,
        "width": width, "height": height,
        "source": source,
    }
    path = out_dir / "intrinsics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Intrinsics ({source}): fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} -> {path}")


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    (out / "color").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    # --- DepthAI pipeline ---
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    monoL = pipeline.createMonoCamera()
    monoR = pipeline.createMonoCamera()
    monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo = pipeline.createStereoDepth()
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)  # align depth to RGB
    stereo.setOutputSize(DEPTH_WIDTH, DEPTH_HEIGHT)   # keep it light for first run

    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    cam.video.link(xoutRgb.input)

    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName("depth")
    stereo.depth.link(xoutDepth.input)  # 16-bit depth in millimetres

    print(f"Starting capture -> {out}/")
    print("Press 'q' to quit.")

    with dai.Device(pipeline) as device:
        # Save real camera intrinsics (or approximate fallback)
        save_intrinsics(device, out, DEPTH_WIDTH, DEPTH_HEIGHT)

        qRgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        qDepth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        with open(out / "timestamps.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "t_wall_s", "t_device_ms"])

            idx = 0
            while True:
                inRgb = qRgb.tryGet()
                inDepth = qDepth.tryGet()

                if inRgb is None or inDepth is None:
                    time.sleep(0.001)
                    continue

                rgb = inRgb.getCvFrame()                 # BGR uint8
                depth = inDepth.getFrame()               # uint16 depth in mm

                # Save files
                color_path = out / "color" / f"color_{idx:06d}.jpg"
                depth_path = out / "depth" / f"depth_{idx:06d}.png"

                cv2.imwrite(str(color_path), rgb)

                # Depth saved as 16-bit PNG (mm)
                cv2.imwrite(str(depth_path), depth)

                t_wall = time.time()
                t_dev = inRgb.getTimestampDevice().total_seconds() * 1000.0
                w.writerow([idx, t_wall, int(t_dev)])

                # Preview (optional)
                if not args.no_preview:
                    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imshow("rgb", rgb)
                    cv2.imshow("depth_vis", depth_vis)

                idx += 1
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    print(f"Done. {idx} frames saved to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
