#!/usr/bin/env python3
"""
FluxSpace Viewer (Mesh + Heatmap)
- Loads:
  - processed/open3d_mesh.ply (surface mesh)
  - exports/volume.npz        (voxel grid heat volume)
- UI:
  - Toggle mesh
  - Toggle heatmap
  - Threshold slider (keeps voxels with value >= threshold)
  - Opacity slider
  - Reframe button (robust percentile bbox to avoid "tiny dot" view)

Run:
  python3 pipelines/3d/view_scan_toggle.py --run "$RUN_DIR"

Notes:
- This viewer intentionally uses a ROBUST bounding box (percentile clamp)
  to avoid outliers making the view useless.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Run directory, e.g. data/runs/run_YYYYMMDD_HHMM")
    p.add_argument("--mesh", default="processed/open3d_mesh.ply", help="Path relative to run dir")
    p.add_argument("--volume", default="exports/volume.npz", help="Path relative to run dir")
    p.add_argument("--title", default="FluxSpace Viewer (Mesh + Heatmap)")
    p.add_argument("--percentile-lo", type=float, default=1.0, help="robust bbox low percentile")
    p.add_argument("--percentile-hi", type=float, default=99.0, help="robust bbox high percentile")
    return p.parse_args()


def robust_bbox_from_points(pts_np: np.ndarray, lo=1.0, hi=99.0) -> o3d.geometry.AxisAlignedBoundingBox:
    """
    Returns a robust AABB based on percentile clamping (good against outliers).
    """
    if pts_np is None or pts_np.size == 0:
        return o3d.geometry.AxisAlignedBoundingBox(np.array([-1, -1, -1]), np.array([1, 1, 1]))

    pts = np.asarray(pts_np, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {pts.shape}")

    p_lo = np.percentile(pts, lo, axis=0)
    p_hi = np.percentile(pts, hi, axis=0)

    # padding proportional to bbox diagonal
    diag = np.linalg.norm(p_hi - p_lo)
    pad = 0.02 * diag + 1e-6
    return o3d.geometry.AxisAlignedBoundingBox(p_lo - pad, p_hi + pad)


def load_volume_npz(path: Path):
    d = np.load(str(path), allow_pickle=True)
    keys = list(d.files)

    # Try common key names
    vol_key = None
    for k in ("volume", "V", "grid", "values"):
        if k in keys:
            vol_key = k
            break
    if vol_key is None:
        # fall back to first key
        vol_key = keys[0]

    V = d[vol_key]

    origin = None
    voxel_size = None

    if "origin" in keys:
        origin = np.array(d["origin"], dtype=np.float64).reshape(3)
    elif "origin_m" in keys:
        origin = np.array(d["origin_m"], dtype=np.float64).reshape(3)

    if "voxel_size" in keys:
        voxel_size = float(d["voxel_size"])
    elif "voxel_size_m" in keys:
        voxel_size = float(d["voxel_size_m"])
    elif "vs" in keys:
        voxel_size = float(d["vs"])

    if origin is None or voxel_size is None:
        raise ValueError(
            f"{path.name} missing required fields. Found keys={keys}. "
            "Need 'origin' (3,) and 'voxel_size' (float) plus a volume array."
        )

    return V, origin, voxel_size, keys


class FluxSpaceViewer:
    def __init__(
        self,
        run_dir: Path,
        mesh_rel: str,
        volume_rel: str,
        title: str,
        pct_lo: float,
        pct_hi: float,
    ):
        self.run_dir = run_dir
        self.mesh_path = run_dir / mesh_rel
        self.vol_path = run_dir / volume_rel
        self.title = title
        self.pct_lo = pct_lo
        self.pct_hi = pct_hi

        # State
        self.mesh_visible = True
        self.heat_visible = True
        self.default_thresh = None  # set after loading volume
        self.heat_threshold = None
        self.heat_opacity = 0.35

        # Geometry
        self.mesh = None  # TriangleMesh
        self.V = None  # 3D numpy array
        self.origin = None  # (3,)
        self.voxel_size = None  # float
        self.heat_geom = None  # PointCloud

        # Render names
        self.MESH_NAME = "surface_mesh"
        self.HEAT_NAME = "heatmap_voxels"

        # Materials
        self.mesh_mat = rendering.MaterialRecord()
        self.mesh_mat.shader = "defaultLit"  # mesh
        self.mesh_mat.base_color = (0.85, 0.85, 0.85, 1.0)

        self.heat_mat = rendering.MaterialRecord()
        self.heat_mat.shader = "defaultUnlit"  # colored points
        self.heat_mat.point_size = 3.0

        # Build UI
        self._build_app()

    def _build_app(self):
        self.app = gui.Application.instance
        self.app.initialize()

        self.window = self.app.create_window(self.title, 1400, 900)
        self.window.set_on_layout(self._on_layout)

        em = self.window.theme.font_size
        self.margin = int(0.5 * em)
        self.panel_width = int(18 * em)

        # Left panel
        self.panel = gui.Vert(0, gui.Margins(self.margin, self.margin, self.margin, self.margin))

        self.chk_mesh = gui.Checkbox("Show surface mesh")
        self.chk_mesh.checked = True
        self.chk_mesh.set_on_checked(self._on_toggle_mesh)
        self.panel.add_child(self.chk_mesh)

        self.chk_heat = gui.Checkbox("Show heatmap")
        self.chk_heat.checked = True
        self.chk_heat.set_on_checked(self._on_toggle_heat)
        self.panel.add_child(self.chk_heat)

        self.panel.add_child(gui.Label("Heat threshold (keeps\nvoxels with value >=\nthreshold)"))
        self.thresh_slider = gui.Slider(gui.Slider.DOUBLE)
        self.thresh_slider.set_limits(0.0, 1.0)
        self.thresh_slider.double_value = 0.5
        self.thresh_slider.set_on_value_changed(self._on_thresh_changed)
        self.panel.add_child(self.thresh_slider)

        self.panel.add_child(gui.Label("Heat opacity"))
        self.opacity_slider = gui.Slider(gui.Slider.DOUBLE)
        self.opacity_slider.set_limits(0.05, 1.0)
        self.opacity_slider.double_value = self.heat_opacity
        self.opacity_slider.set_on_value_changed(self._on_opacity_changed)
        self.panel.add_child(self.opacity_slider)

        self.panel.add_child(gui.Label(" "))
        self.btn_reframe = gui.Button("Reframe")
        self.btn_reframe.set_on_clicked(self._on_reframe)
        self.panel.add_child(self.btn_reframe)

        # Scene widget
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([1, 1, 1, 1])  # white

        self.window.add_child(self.panel)
        self.window.add_child(self.scene_widget)

        # Load data & add geometry
        self._load_all()
        self._add_initial_geometries()
        self._fit_camera()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.panel.frame = gui.Rect(r.x, r.y, self.panel_width, r.height)
        self.scene_widget.frame = gui.Rect(r.x + self.panel_width, r.y, r.width - self.panel_width, r.height)

    def _load_all(self):
        if not self.mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {self.mesh_path}")
        if not self.vol_path.exists():
            raise FileNotFoundError(f"Volume not found: {self.vol_path}")

        # Mesh
        self.mesh = o3d.io.read_triangle_mesh(str(self.mesh_path))
        if self.mesh.is_empty():
            raise ValueError(f"Mesh loaded but empty: {self.mesh_path}")
        if not self.mesh.has_vertex_normals():
            self.mesh.compute_vertex_normals()

        # Volume
        self.V, self.origin, self.voxel_size, keys = load_volume_npz(self.vol_path)
        self.V = np.asarray(self.V)

        vmin, vmax = float(np.min(self.V)), float(np.max(self.V))
        # Pick a sensible default threshold: near top 90th percentile (keeps "hot" voxels)
        self.default_thresh = float(np.percentile(self.V, 90.0))
        self.heat_threshold = self.default_thresh

        # Configure slider range and initial value
        # Range: [vmin, vmax], clamp if degenerate
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1.0
        self.thresh_slider.set_limits(vmin, vmax)
        self.thresh_slider.double_value = self.default_thresh

    def _add_initial_geometries(self):
        # Add mesh
        self.scene_widget.scene.add_geometry(self.MESH_NAME, self.mesh, self.mesh_mat)

        # Add heat
        self._add_or_update_heat()
        if self.heat_geom is not None:
            self.scene_widget.scene.add_geometry(self.HEAT_NAME, self.heat_geom, self.heat_mat)

    def _make_heat_pointcloud(self, thresh: float) -> o3d.geometry.PointCloud:
        idx = np.argwhere(self.V >= thresh)
        if idx.shape[0] == 0:
            return o3d.geometry.PointCloud()

        # voxel centers -> world points
        pts = self.origin + (idx + 0.5) * float(self.voxel_size)

        # values for colors
        vals = self.V[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float64)

        # normalize 0..1 for coloring
        v0 = float(np.min(vals))
        v1 = float(np.max(vals))
        if abs(v1 - v0) < 1e-12:
            t = np.zeros_like(vals)
        else:
            t = (vals - v0) / (v1 - v0)

        # simple colormap-ish (blue->green->yellow)
        # (keep it lightweight; no matplotlib dependency)
        colors = np.zeros((t.shape[0], 3), dtype=np.float64)
        colors[:, 0] = np.clip(2 * (t - 0.5), 0, 1)          # R grows in top half
        colors[:, 1] = np.clip(2 * (1 - np.abs(t - 0.5)), 0, 1)  # G peaks mid
        colors[:, 2] = np.clip(1 - 2 * t, 0, 1)              # B fades out

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def _add_or_update_heat(self):
        self.heat_threshold = float(self.thresh_slider.double_value)

        self.heat_geom = self._make_heat_pointcloud(self.heat_threshold)

        # update opacity via material alpha
        self.heat_mat.base_color = (1.0, 1.0, 1.0, float(self.opacity_slider.double_value))

        # If already in scene, remove & re-add (safe + simple)
        if self.scene_widget.scene.has_geometry(self.HEAT_NAME):
            self.scene_widget.scene.remove_geometry(self.HEAT_NAME)

        if self.heat_visible and self.heat_geom is not None and not self.heat_geom.is_empty():
            self.scene_widget.scene.add_geometry(self.HEAT_NAME, self.heat_geom, self.heat_mat)

    def _fit_camera(self):
        boxes = []

        # mesh bbox
        if self.mesh is not None:
            try:
                v = np.asarray(self.mesh.vertices)
                boxes.append(robust_bbox_from_points(v, self.pct_lo, self.pct_hi))
            except Exception:
                boxes.append(self.mesh.get_axis_aligned_bounding_box())

        # heat bbox (based on currently visible threshold)
        if self.heat_visible and self.V is not None:
            thresh = float(self.thresh_slider.double_value)
            idx = np.argwhere(self.V >= thresh)
            if idx.shape[0] > 0:
                pts = self.origin + (idx + 0.5) * float(self.voxel_size)
                boxes.append(robust_bbox_from_points(pts, self.pct_lo, self.pct_hi))

        if not boxes:
            bbox = o3d.geometry.AxisAlignedBoundingBox(np.array([-1, -1, -1]), np.array([1, 1, 1]))
        else:
            bbox = boxes[0]
            for b in boxes[1:]:
                bbox += b  # union

        self.scene_widget.scene.setup_camera(60.0, bbox, bbox.get_center())

    # ---------------- UI callbacks ----------------

    def _on_toggle_mesh(self, checked: bool):
        self.mesh_visible = bool(checked)
        if self.scene_widget.scene.has_geometry(self.MESH_NAME):
            self.scene_widget.scene.remove_geometry(self.MESH_NAME)
        if self.mesh_visible:
            self.scene_widget.scene.add_geometry(self.MESH_NAME, self.mesh, self.mesh_mat)
        self._fit_camera()

    def _on_toggle_heat(self, checked: bool):
        self.heat_visible = bool(checked)
        if self.scene_widget.scene.has_geometry(self.HEAT_NAME):
            self.scene_widget.scene.remove_geometry(self.HEAT_NAME)
        if self.heat_visible:
            self._add_or_update_heat()
        self._fit_camera()

    def _on_thresh_changed(self, _value: float):
        if self.heat_visible:
            self._add_or_update_heat()
        self._fit_camera()

    def _on_opacity_changed(self, value: float):
        self.heat_opacity = float(value)
        if self.heat_visible:
            self._add_or_update_heat()

    def _on_reframe(self):
        self._fit_camera()

    def run(self):
        self.app.run()


def main():
    args = parse_args()
    run_dir = Path(args.run).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    viewer = FluxSpaceViewer(
        run_dir=run_dir,
        mesh_rel=args.mesh,
        volume_rel=args.volume,
        title=args.title,
        pct_lo=args.percentile_lo,
        pct_hi=args.percentile_hi,
    )
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
