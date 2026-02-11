import glob
from pathlib import Path
import numpy as np
import open3d as o3d

BASE = Path("oak_capture")
color_files = sorted(glob.glob(str(BASE / "color" / "color_*.jpg")))
depth_files = sorted(glob.glob(str(BASE / "depth" / "depth_*.png")))

assert len(color_files) == len(depth_files), "Color/depth count mismatch"

# NOTE: These intrinsics are not calibrated. For a first demo, use a rough pinhole model.
# Better later: read calibration from DepthAI and build exact intrinsics.
width, height = 640, 400
fx = fy = 600.0
cx = width / 2.0
cy = height / 2.0
intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.01,  # 1 cm voxels (increase if slow)
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

# Simple frame-to-frame odometry integration (baseline demo)
prev_rgbd = None
prev_pose = np.eye(4)
poses = [prev_pose.copy()]

for i, (cf, df) in enumerate(zip(color_files, depth_files)):
    color = o3d.io.read_image(cf)
    depth = o3d.io.read_image(df)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,
        depth_scale=1000.0,     # mm -> meters
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )

    if prev_rgbd is not None:
        option = o3d.pipelines.odometry.OdometryOption()
        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            prev_rgbd, rgbd, intr, np.eye(4),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option
        )
        if success:
            prev_pose = prev_pose @ trans
        # if it fails, keep previous pose (demo-friendly)

    poses.append(prev_pose.copy())
    volume.integrate(rgbd, intr, np.linalg.inv(prev_pose))
    prev_rgbd = rgbd

    if i % 30 == 0:
        print(f"Integrated frame {i}/{len(color_files)}")

mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()

out = Path("oak_capture") / "open3d_mesh.ply"
o3d.io.write_triangle_mesh(str(out), mesh)
print("Wrote:", out)

o3d.visualization.draw_geometries([mesh])
