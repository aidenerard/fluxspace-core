#!/usr/bin/env python3
"""
capture_oak_rgbd_vio.py

Capture synchronized RGB + aligned depth from an OAK-D Lite with IMU-based
visual-inertial odometry (VIO), producing a device-side trajectory.

Trajectory estimation combines:
  - IMU orientation from gyroscope integration with accelerometer tilt
    correction (complementary filter on the host)
  - Translation from optical-flow feature tracking + depth unprojection,
    solved given the known rotation from the IMU

Outputs (same layout as capture_oak_rgbd.py + trajectory):
  --out/
    color/color_000000.jpg
    depth/depth_000000.png   (uint16, mm)
    timestamps.csv           (idx, t_wall_s, t_rgb_dev_ms, t_depth_dev_ms)
    intrinsics.json          (fx, fy, cx, cy, width, height)
    trajectory_device.csv    (t_rel_s, x, y, z, qx, qy, qz, qw)
    imu_raw.csv              (optional, with --save-imu)

Press 'q' in the preview window (or Ctrl+C) to stop.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import depthai as dai

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAP_W, CAP_H = 640, 400
IMU_RATE = 200
DEPTH_MIN_M = 0.10
DEPTH_MAX_M = 3.0
FLOW_MAX_CORNERS = 500
FLOW_QUALITY = 0.01
FLOW_MIN_DIST = 10

# ---------------------------------------------------------------------------
# Quaternion helpers — convention: [x, y, z, w] (Hamilton)
# ---------------------------------------------------------------------------

def _qnorm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _qconj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _qfrom_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    ha = angle * 0.5
    s = np.sin(ha)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(ha)])


def _qrotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector *v* by quaternion *q*."""
    vq = np.array([v[0], v[1], v[2], 0.0])
    return _qmul(_qmul(q, vq), _qconj(q))[:3]


def _qto_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _qslerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        return _qnorm(q0 + t * (q1 - q0))
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion rotating unit-vector *a* onto unit-vector *b*."""
    cross = np.cross(a, b)
    dot = float(np.dot(a, b))
    if np.linalg.norm(cross) < 1e-8:
        if dot > 0:
            return np.array([0.0, 0.0, 0.0, 1.0])
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = np.cross(a, perp)
        perp /= np.linalg.norm(perp)
        return np.array([perp[0], perp[1], perp[2], 0.0])
    w = 1.0 + dot
    return _qnorm(np.array([cross[0], cross[1], cross[2], w]))

# ---------------------------------------------------------------------------
# IMU orientation tracker  (complementary filter)
# ---------------------------------------------------------------------------

class IMUTracker:
    """Integrate gyroscope for orientation, correct pitch/roll with accelerometer."""

    def __init__(self, alpha: float = 0.98):
        self.q = np.array([0.0, 0.0, 0.0, 1.0])
        self.alpha = alpha
        self._gravity_ref: np.ndarray | None = None
        self._last_ts: float | None = None

    def update(self, gyro: np.ndarray, accel: np.ndarray, ts_s: float) -> np.ndarray:
        if self._last_ts is None:
            a_norm = np.linalg.norm(accel)
            if a_norm > 1e-3:
                self._gravity_ref = accel / a_norm
            else:
                self._gravity_ref = np.array([0.0, 0.0, -1.0])
            self._last_ts = ts_s
            return self.q.copy()

        dt = ts_s - self._last_ts
        self._last_ts = ts_s
        if dt <= 0 or dt > 0.1:
            return self.q.copy()

        # Gyroscope integration
        w = np.asarray(gyro, dtype=np.float64)
        angle = np.linalg.norm(w) * dt
        if angle > 1e-8:
            axis = w / np.linalg.norm(w)
            dq = _qfrom_axis_angle(axis, angle)
            q_gyro = _qnorm(_qmul(self.q, dq))
        else:
            q_gyro = self.q.copy()

        # Accelerometer tilt correction
        a_norm = np.linalg.norm(accel)
        if 7.0 < a_norm < 13.0 and self._gravity_ref is not None:
            g_meas = accel / a_norm
            g_pred = _qrotate(_qconj(q_gyro), self._gravity_ref)
            correction = _rotation_between(g_pred, g_meas)
            correction_small = _qslerp(
                np.array([0.0, 0.0, 0.0, 1.0]), correction, 1.0 - self.alpha
            )
            self.q = _qnorm(_qmul(q_gyro, correction_small))
        else:
            self.q = q_gyro

        return self.q.copy()

# ---------------------------------------------------------------------------
# Translation estimator  (optical flow + depth)
# ---------------------------------------------------------------------------

class TranslationEstimator:
    """Estimate frame-to-frame translation from tracked depth features."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self._prev_gray: np.ndarray | None = None
        self._prev_depth: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._lk_params = dict(winSize=(21, 21), maxLevel=3,
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def _unproject(self, pts2d: np.ndarray, depth_m: np.ndarray):
        h, w = depth_m.shape
        u = pts2d[:, 0]
        v = pts2d[:, 1]
        ui = np.clip(np.round(u).astype(int), 0, w - 1)
        vi = np.clip(np.round(v).astype(int), 0, h - 1)
        d = depth_m[vi, ui]
        valid = (d > DEPTH_MIN_M) & (d < DEPTH_MAX_M) & np.isfinite(d)
        X = (u - self.cx) * d / self.fx
        Y = (v - self.cy) * d / self.fy
        return np.column_stack([X, Y, d]), valid

    def _detect(self, gray: np.ndarray):
        pts = cv2.goodFeaturesToTrack(gray, FLOW_MAX_CORNERS, FLOW_QUALITY, FLOW_MIN_DIST)
        return pts if pts is not None else np.empty((0, 1, 2), dtype=np.float32)

    def estimate(self, gray: np.ndarray, depth_m: np.ndarray,
                 R_prev: np.ndarray, R_curr: np.ndarray) -> np.ndarray | None:
        """Return world-frame translation delta, or None if insufficient data."""
        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            self._prev_depth = depth_m.copy()
            self._prev_pts = self._detect(gray)
            return None

        if self._prev_pts is None or len(self._prev_pts) < 10:
            self._prev_pts = self._detect(gray)
            self._prev_gray = gray.copy()
            self._prev_depth = depth_m.copy()
            return None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **self._lk_params
        )
        if curr_pts is None:
            self._prev_gray = gray.copy()
            self._prev_depth = depth_m.copy()
            self._prev_pts = self._detect(gray)
            return None

        good = status.ravel() == 1
        if good.sum() < 5:
            self._prev_gray = gray.copy()
            self._prev_depth = depth_m.copy()
            self._prev_pts = self._detect(gray)
            return None

        prev_good = self._prev_pts[good].reshape(-1, 2)
        curr_good = curr_pts[good].reshape(-1, 2)

        p3d_prev, v_prev = self._unproject(prev_good, self._prev_depth)
        p3d_curr, v_curr = self._unproject(curr_good, depth_m)
        both = v_prev & v_curr
        if both.sum() < 5:
            self._prev_gray = gray.copy()
            self._prev_depth = depth_m.copy()
            self._prev_pts = self._detect(gray)
            return None

        pp = p3d_prev[both]
        pc = p3d_curr[both]

        # World-frame positions: P_w = R @ P_cam + t
        # Same point: R_prev @ pp + t_prev = R_curr @ pc + t_curr
        # dt = R_prev @ pp - R_curr @ pc
        wp = (R_prev @ pp.T).T
        wc = (R_curr @ pc.T).T
        dt_samples = wp - wc

        dt = np.median(dt_samples, axis=0)
        if np.any(np.std(dt_samples, axis=0) > 0.5):
            dt = np.zeros(3)

        self._prev_gray = gray.copy()
        self._prev_depth = depth_m.copy()
        self._prev_pts = self._detect(gray)
        return dt

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture OAK RGB+D with IMU VIO trajectory"
    )
    p.add_argument("--out", required=True,
                   help="Output directory, e.g. $RUN_DIR/raw/oak_rgbd")
    p.add_argument("--fps", type=int, default=15, help="Camera FPS (default: 15)")
    p.add_argument("--no-preview", action="store_true",
                   help="Disable OpenCV preview windows")
    p.add_argument("--save-imu", action="store_true",
                   help="Write raw IMU readings to imu_raw.csv")
    p.add_argument("--imu-alpha", type=float, default=0.98,
                   help="Complementary filter gyro trust (0-1, default: 0.98)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Intrinsics helper
# ---------------------------------------------------------------------------

def _save_intrinsics(device: dai.Device, out: Path, w: int, h: int) -> dict:
    fx = fy = 600.0
    cx, cy = w / 2.0, h / 2.0
    source = "approximate"
    try:
        calib = device.readCalibration()
        M = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, w, h)
        fx, fy, cx, cy = float(M[0][0]), float(M[1][1]), float(M[0][2]), float(M[1][2])
        source = "oak_calibration"
    except Exception as e:
        print(f"WARNING: Could not read calibration ({e}); using approximate intrinsics.")
    data = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": w, "height": h, "source": source}
    (out / "intrinsics.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  Intrinsics ({source}): fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    return data

# ---------------------------------------------------------------------------
# DepthAI pipeline builder
# ---------------------------------------------------------------------------

def _build_pipeline(fps: int, enable_imu: bool) -> dai.Pipeline:
    pipeline = dai.Pipeline()

    # RGB
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    if hasattr(cam, "setVideoSize"):
        cam.setVideoSize(CAP_W, CAP_H)
    if hasattr(cam, "setFps"):
        cam.setFps(fps)

    # Stereo depth
    monoL = pipeline.create(dai.node.MonoCamera)
    monoR = pipeline.create(dai.node.MonoCamera)
    monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.setOutputSize(CAP_W, CAP_H)
    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam.video.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    if enable_imu:
        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, IMU_RATE)
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, IMU_RATE)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)
        xout_imu = pipeline.create(dai.node.XLinkOut)
        xout_imu.setStreamName("imu")
        imu.out.link(xout_imu.input)

    return pipeline

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_shutdown = False

def _sig_handler(sig, frame):
    global _shutdown
    _shutdown = True

def main() -> int:
    global _shutdown
    args = parse_args()
    out = Path(args.out)
    (out / "color").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, _sig_handler)

    # Try pipeline with IMU; fall back without
    have_imu = True
    pipeline = _build_pipeline(args.fps, enable_imu=True)
    try:
        device = dai.Device(pipeline)
    except Exception as exc:
        print(f"WARNING: Pipeline with IMU failed ({exc}); retrying without IMU.")
        have_imu = False
        pipeline = _build_pipeline(args.fps, enable_imu=False)
        device = dai.Device(pipeline)

    if have_imu:
        try:
            imu_type = device.getConnectedIMU()
            print(f"  IMU detected: {imu_type}")
        except Exception:
            pass

    with device:
        intr = _save_intrinsics(device, out, CAP_W, CAP_H)
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["cx"], intr["cy"]

        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
        q_imu = None
        if have_imu:
            try:
                q_imu = device.getOutputQueue("imu", maxSize=50, blocking=False)
            except Exception:
                have_imu = False
                print("WARNING: Could not open IMU queue; proceeding without IMU.")

        tracker = IMUTracker(alpha=args.imu_alpha) if have_imu else None
        estimator = TranslationEstimator(fx, fy, cx, cy)

        trajectory: list[tuple[float, np.ndarray, np.ndarray]] = []
        # Each entry: (t_rel_s, position_xyz, orientation_quat_xyzw)

        position = np.zeros(3, dtype=np.float64)
        orientation = np.array([0.0, 0.0, 0.0, 1.0])
        prev_R = np.eye(3)

        imu_csv_f = None
        imu_csv_w = None
        if args.save_imu and have_imu:
            imu_csv_f = open(out / "imu_raw.csv", "w", newline="")
            imu_csv_w = csv.writer(imu_csv_f)
            imu_csv_w.writerow(["t_dev_s", "ax", "ay", "az", "gx", "gy", "gz"])

        mode_label = "VIO (IMU + depth)" if have_imu else "depth-only (no IMU)"
        print(f"Starting capture -> {out}  [{mode_label}]")
        print("Press 'q' in preview or Ctrl+C to stop.")

        ts_f = open(out / "timestamps.csv", "w", newline="")
        ts_w = csv.writer(ts_f)
        ts_w.writerow(["idx", "t_wall_s", "t_rgb_dev_ms", "t_depth_dev_ms"])

        idx = 0
        t0_dev: float | None = None

        try:
            while not _shutdown:
                # --- Drain IMU queue ---
                if q_imu is not None:
                    try:
                        imu_data = q_imu.tryGet()
                        if imu_data is not None:
                            for pkt in imu_data.packets:
                                a = pkt.acceleroMeter
                                g = pkt.gyroscope
                                ts_a = a.getTimestampDevice().total_seconds()
                                accel = np.array([a.x, a.y, a.z], dtype=np.float64)
                                gyro = np.array([g.x, g.y, g.z], dtype=np.float64)
                                if tracker is not None:
                                    orientation = tracker.update(gyro, accel, ts_a)
                                if imu_csv_w is not None:
                                    imu_csv_w.writerow([
                                        f"{ts_a:.6f}",
                                        f"{a.x:.6f}", f"{a.y:.6f}", f"{a.z:.6f}",
                                        f"{g.x:.6f}", f"{g.y:.6f}", f"{g.z:.6f}",
                                    ])
                    except Exception:
                        pass

                # --- Get frame pair ---
                in_rgb = q_rgb.tryGet()
                in_depth = q_depth.tryGet()
                if in_rgb is None or in_depth is None:
                    time.sleep(0.001)
                    continue

                rgb = in_rgb.getCvFrame()
                depth_u16 = in_depth.getFrame()

                # Timestamps
                t_wall = time.time()
                t_rgb_dev = in_rgb.getTimestampDevice().total_seconds()
                t_dep_dev = in_depth.getTimestampDevice().total_seconds()
                if t0_dev is None:
                    t0_dev = t_rgb_dev
                t_rel = t_rgb_dev - t0_dev

                # Save images
                cv2.imwrite(str(out / "color" / f"color_{idx:06d}.jpg"), rgb)
                cv2.imwrite(str(out / "depth" / f"depth_{idx:06d}.png"), depth_u16)
                ts_w.writerow([
                    idx, t_wall,
                    int(t_rgb_dev * 1000.0),
                    int(t_dep_dev * 1000.0),
                ])

                # --- Translation estimation ---
                R_curr = _qto_rotmat(orientation)
                depth_m = depth_u16.astype(np.float64) * 0.001

                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if len(rgb.shape) == 3 else rgb.copy()

                dt = estimator.estimate(gray, depth_m, prev_R, R_curr)
                if dt is not None and np.isfinite(dt).all():
                    position = position + dt
                prev_R = R_curr.copy()

                # Record pose
                if np.isfinite(position).all() and np.isfinite(orientation).all():
                    trajectory.append((t_rel, position.copy(), orientation.copy()))
                else:
                    trajectory.append((t_rel, np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])))

                # Preview
                if not args.no_preview:
                    depth_vis = cv2.normalize(depth_u16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imshow("rgb", rgb)
                    cv2.imshow("depth", depth_vis)

                idx += 1
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

        finally:
            ts_f.close()
            if imu_csv_f is not None:
                imu_csv_f.close()

    # --- Write trajectory ---
    traj_path = out / "trajectory_device.csv"
    n_valid = 0
    with open(traj_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_rel_s", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for t_rel, pos, quat in trajectory:
            vals = list(pos) + list(quat)
            if not np.isfinite(vals).all():
                continue
            w.writerow([
                f"{t_rel:.6f}",
                f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}",
                f"{quat[0]:.6f}", f"{quat[1]:.6f}", f"{quat[2]:.6f}", f"{quat[3]:.6f}",
            ])
            n_valid += 1

    print(f"\nDone. Saved {idx} frames -> {out}")
    print(f"  trajectory_device.csv : {n_valid} poses ({len(trajectory) - n_valid} invalid skipped)")
    if have_imu:
        print("  Pose source: IMU (gyro+accel) + depth optical flow")
    else:
        print("  Pose source: depth optical flow only (no IMU detected)")
        print("  TIP: Without IMU, rotations will be identity. Consider using")
        print("       --pose-source odom in open3d_reconstruct.py instead.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
