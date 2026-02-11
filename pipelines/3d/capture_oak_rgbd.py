import os, time, csv
from pathlib import Path

import cv2
import numpy as np
import depthai as dai

OUT = Path("oak_capture")
(OUT / "color").mkdir(parents=True, exist_ok=True)
(OUT / "depth").mkdir(parents=True, exist_ok=True)

# --- DepthAI pipeline ---
pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
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
stereo.setOutputSize(640, 400)  # keep it light for first run

monoL.out.link(stereo.left)
monoR.out.link(stereo.right)

xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
cam.video.link(xoutRgb.input)

xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)  # 16-bit depth in millimeters

print("Starting capture. Press 'q' to quit.")
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue("depth", maxSize=4, blocking=False)

    with open(OUT / "timestamps.csv", "w", newline="") as f:
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
            color_path = OUT / "color" / f"color_{idx:06d}.jpg"
            depth_path = OUT / "depth" / f"depth_{idx:06d}.png"

            cv2.imwrite(str(color_path), rgb)

            # Depth saved as 16-bit PNG (mm)
            cv2.imwrite(str(depth_path), depth)

            t_wall = time.time()
            t_dev = inRgb.getTimestampDevice().total_seconds() * 1000.0
            w.writerow([idx, t_wall, int(t_dev)])

            # Preview (scaled depth for display)
            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow("rgb", rgb)
            cv2.imshow("depth_vis", depth_vis)

            idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

print("Done. Saved to oak_capture/")
