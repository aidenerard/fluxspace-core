#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, time
from pathlib import Path

import cv2
import numpy as np
import depthai as dai

DEFAULT_OUT = "oak_capture"
CAP_W, CAP_H = 640, 400

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--fps", type=int, default=15)
    return p.parse_args()

def save_intrinsics(device: dai.Device, out_dir: Path, w: int, h: int):
    fx = fy = 600.0
    cx, cy = w/2.0, h/2.0
    source = "approximate"
    try:
        calib = device.readCalibration()
        M = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, w, h)
        fx, fy, cx, cy = float(M[0][0]), float(M[1][1]), float(M[0][2]), float(M[1][2])
        source = "oak_calibration"
    except Exception as e:
        print(f"WARNING: intrinsics fallback ({e})")

    (out_dir / "intrinsics.json").write_text(json.dumps({
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "width": w, "height": h, "source": source
    }, indent=2), encoding="utf-8")

def main():
    args = parse_args()
    out = Path(args.out)
    (out / "color").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    pipeline = dai.Pipeline()

    # Use new Camera node if available; otherwise fall back to legacy nodes
    Camera = getattr(dai.node, "Camera", None)

    if Camera is not None:
        cam = pipeline.create(Camera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setFps(args.fps)

        # Request RGB + depth from the same Camera node
        rgb = cam.createOutput("rgb")          # color stream
        depth = cam.createOutput("depth")      # aligned depth stream (if supported)

        # Ensure sizes
        camRgb = cam.getRgb()
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        if hasattr(camRgb, "setVideoSize"):
            camRgb.setVideoSize(CAP_W, CAP_H)
    else:
        # Legacy fallback (works if your build still has these)
        color = pipeline.create(dai.node.ColorCamera)
        color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        color.setInterleaved(False)
        color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        if hasattr(color, "setVideoSize"):
            color.setVideoSize(CAP_W, CAP_H)

        monoL = pipeline.create(dai.node.MonoCamera)
        monoR = pipeline.create(dai.node.MonoCamera)
        monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        stereo = pipeline.create(dai.node.StereoDepth)
        preset = getattr(dai.node.StereoDepth.PresetMode, "HIGH_DENSITY", dai.node.StereoDepth.PresetMode.FAST_DENSITY)
        stereo.setDefaultProfilePreset(preset)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(CAP_W, CAP_H)

        monoL.out.link(stereo.left)
        monoR.out.link(stereo.right)

        # IMPORTANT: instead of XLinkOut, use stream names via device queues (see below)
        rgb = color.video
        depth = stereo.depth

    print(f"Starting capture -> {out} (press q to quit)")

    with dai.Device(pipeline) as device:
        save_intrinsics(device, out, CAP_W, CAP_H)

        # Queue names vary across builds. Try common ones.
        # If these fail, we'll print available streams.
        def getq(name):
            return device.getOutputQueue(name, maxSize=4, blocking=False)

        # Try to auto-detect stream names
        available = []
        try:
            available = device.getOutputQueueNames()
        except Exception:
            pass

        # Common names in many DepthAI builds
        rgb_names = ["rgb", "color", "video"]
        depth_names = ["depth", "disparityDepth", "depth_raw", "stereoDepth"]

        qRgb = None
        qDepth = None
        for n in rgb_names:
            try:
                qRgb = getq(n); break
            except Exception:
                pass
        for n in depth_names:
            try:
                qDepth = getq(n); break
            except Exception:
                pass

        if qRgb is None or qDepth is None:
            # Print what exists so we can pick the right names
            print("Could not open queues. Available output queues:", available)
            print("Fix: pick the correct names from above and hardcode them.")
            return 1

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

                rgb_frame = inRgb.getCvFrame() if hasattr(inRgb, "getCvFrame") else None
                if rgb_frame is None:
                    # Some builds deliver ImgFrame without getCvFrame
                    rgb_frame = inRgb.getFrame()

                depth_frame = inDepth.getFrame()

                cv2.imwrite(str(out / "color" / f"color_{idx:06d}.jpg"), rgb_frame)
                cv2.imwrite(str(out / "depth" / f"depth_{idx:06d}.png"), depth_frame)

                t_wall = time.time()
                t_rgb = int(inRgb.getTimestampDevice().total_seconds() * 1000.0)
                t_dep = int(inDepth.getTimestampDevice().total_seconds() * 1000.0)
                w.writerow([idx, t_wall, t_rgb, t_dep])

                if not args.no_preview:
                    depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imshow("rgb", rgb_frame)
                    cv2.imshow("depth_vis", depth_vis)

                idx += 1
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
