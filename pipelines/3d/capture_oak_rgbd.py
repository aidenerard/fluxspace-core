#!/usr/bin/env python3
"""
capture_oak_rgbd.py (compat)

Record synchronised RGB + aligned depth frames from a Luxonis OAK-D Lite
using the DepthAI SDK, with compatibility across older DepthAI APIs
(e.g. depthai==3.3.0).

Outputs (inside --out directory):
  color/color_000000.jpg   BGR uint8
  depth/depth_000000.png   16-bit depth in millimetres
  timestamps.csv           idx, t_wall_s, t_rgb_dev_ms, t_depth_dev_ms
  intrinsics.json          fx, fy, cx, cy, width, height

Designed to be consumable by open3d_reconstruct.py when --out is $RUN_DIR/raw/oak_rgbd.
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


DEFAULT_OUT = "oak_capture"
# Keep capture light + consistent between RGB and depth
CAP_W = 640
CAP_H = 400


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OAK RGB+aligned depth capture -> frames + timestamps + intrinsics")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"Output directory. Default: {DEFAULT_OUT}")
    p.add_argument("--no-preview", action="store_true", help="Disable OpenCV preview windows")
    p.add_argument("--fps", type=int, default=15, help="Camera FPS (default 15)")
    return p.parse_args()


def _resolve(obj, dotted: str):
    """Resolve dotted attribute path like 'node.ColorCamera' against an object."""
    cur = obj
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def create_node(pipeline: dai.Pipeline, name_candidates: list[str]):
    """
    Create a node, trying several class locations depending on DepthAI version.
    Examples: 'node.ColorCamera' or 'ColorCamera'
    """
    last_exc = None
    for cand in name_candidates:
        cls = _resolve(dai, cand) if "." in cand else getattr(dai, cand, None)
        if cls is None:
            continue
        try:
            return pipeline.create(cls)
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"Could not create node. Tried: {name_candidates}. Last error: {last_exc}")


def save_intrinsics(device: dai.Device, out_dir: Path, width: int, height: int) -> None:
    fx = fy = 600.0
    cx = width / 2.0
    cy = height / 2.0
    source = "approximate"
    try:
        calib = device.readCalibration()
        # Use RGB (CAM_A) intrinsics at the exact capture resolution
        M = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, width, height)
        fx = float(M[0][0])
        fy = float(M[1][1])
        cx = float(M[0][2])
        cy = float(M[1][2])
        source = "oak_calibration"
    except Exception as exc:
        print(f"WARNING: Could not read OAK calibration ({exc}); using approximate intrinsics.")

    data = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height, "source": source}
    path = out_dir / "intrinsics.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Intrinsics ({source}) -> {path}")


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    (out / "color").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    pipeline = dai.Pipeline()

    # Nodes (compatible creation)
    cam = create_node(pipeline, ["node.ColorCamera", "ColorCamera"])
    monoL = create_node(pipeline, ["node.MonoCamera", "MonoCamera"])
    monoR = create_node(pipeline, ["node.MonoCamera", "MonoCamera"])
    stereo = create_node(pipeline, ["node.StereoDepth", "StereoDepth"])

    xoutRgb = create_node(pipeline, ["node.XLinkOut", "XLinkOut"])
    xoutDepth = create_node(pipeline, ["node.XLinkOut", "XLinkOut"])

    # Configure RGB
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    # Force output size to match depth (simplifies reconstruction)
    if hasattr(cam, "setVideoSize"):
        cam.setVideoSize(CAP_W, CAP_H)

    # Configure mono cameras
    monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # StereoDepth settings
    # Use a preset that exists in older DepthAI (FAST_DENSITY works in your version)
    preset = getattr(dai.node.StereoDepth.PresetMode, "HIGH_DENSITY", dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setDefaultProfilePreset(preset)

    # Align depth to RGB (deprecated warnings are OK; still works)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.setOutputSize(CAP_W, CAP_H)

    # Link stereo inputs
    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    # Stream names + links
    xoutRgb.setStreamName("rgb")
    xoutDepth.setStreamName("depth")

    # Use cam.video if available; otherwise cam.preview
    rgb_out = cam.video if hasattr(cam, "video") else cam.preview
    rgb_out.link(xoutRgb.input)
    stereo.depth.link(xoutDepth.input)

    print(f"Starting OAK capture -> {out}/  (press 'q' to quit)")
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

                rgb = inRgb.getCvFrame()
                depth = inDepth.getFrame()  # uint16 mm

                cv2.imwrite(str(out / "color" / f"color_{idx:06d}.jpg"), rgb)
                cv2.imwrite(str(out / "depth" / f"depth_{idx:06d}.png"), depth)

                t_wall = time.time()
                t_rgb_ms = int(inRgb.getTimestampDevice().total_seconds() * 1000.0)
                t_depth_ms = int(inDepth.getTimestampDevice().total_seconds() * 1000.0)
                w.writerow([idx, t_wall, t_rgb_ms, t_depth_ms])

                if not args.no_preview:
                    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imshow("rgb", rgb)
                    cv2.imshow("depth_vis", depth_vis)

                idx += 1
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    print(f"Done. Saved {idx} frames -> {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
