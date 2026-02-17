#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

# Open3D GUI (works on macOS if Open3D installed correctly)
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def load_volume_npz(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    # Expecting keys like: volume/value/grid/origin/voxel_size (your script prints these)
    # We'll be defensive:
    keys = set(d.files)

    # Common patterns
    if "volume" in keys:
        V = d["volume"]
    elif "value" in keys:
        V = d["value"]
    else:
        # fallback: first array
        V = d[d.files[0]]

    origin = d["origin"] if "origin" in keys else np.array([0.0, 0.0, 0.0])
    voxel_size = float(d["voxel_size"]) if "voxel_size" in keys else 0.02

    return V, origin.astype(float), voxel_size


def make_heatmap_pointcloud(V, origin, voxel_size, thresh, max_points=400_000):
    """
    Turn voxels with V >= thresh into a colored point cloud at voxel centers.
    """
    idx = np.argwhere(V >= thresh)
    if idx.shape[0] == 0:
        return o3d.geometry.PointCloud()

    # Downsample if huge
    if idx.shape[0] > max_points:
        step = int(np.ceil(idx.shape[0] / max_points))
        idx = idx[::step]

    # voxel center positions in world coordinates
    pts = origin + (idx + 0.5) * voxel_size

    vals = V[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    denom = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1.0
    t = (vals - vmin) / denom

    # simple colormap (blue->green->yellow-ish)
    # (No matplotlib dependency)
    colors = np.stack([
        np.clip(1.5 * (t - 0.5), 0, 1),          # R
        np.clip(1.5 * (1 - np.abs(t - 0.5)), 0, 1),  # G
        np.clip(1.5 * (0.5 - t), 0, 1)           # B
    ], axis=1).astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


class App:
    def __init__(self, mesh_path: Path, volume_path: Path):
        self.mesh_path = mesh_path
        self.volume_path = volume_path

        self.mesh = None
        self.V = None
        self.origin = None
        self.voxel_size = None

        self.mesh_visible = True
        self.heat_visible = True
        self.alpha = 0.35

        self._load_data()

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("FluxSpace Viewer (Mesh + Heatmap)", 1280, 800)

        # --- Layout
        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin, margin, margin, margin))
        self.window.add_child(self.panel)

        # --- Checkboxes
        self.cb_mesh = gui.Checkbox("Show surface mesh")
        self.cb_mesh.checked = True
        self.cb_mesh.set_on_checked(self._on_toggle_mesh)
        self.panel.add_child(self.cb_mesh)

        self.cb_heat = gui.Checkbox("Show heatmap")
        self.cb_heat.checked = True
        self.cb_heat.set_on_checked(self._on_toggle_heat)
        self.panel.add_child(self.cb_heat)

        # --- Threshold slider
        self.panel.add_child(gui.Label("Heat threshold (keeps voxels with value >= threshold)"))
        self.thresh_slider = gui.Slider(gui.Slider.DOUBLE)
        self.thresh_slider.set_limits(self.vmin, self.vmax)
        self.thresh_slider.double_value = self.default_thresh
        self.thresh_slider.set_on_value_changed(self._on_thresh_changed)
        self.panel.add_child(self.thresh_slider)

        # --- Alpha slider
        self.panel.add_child(gui.Label("Heat opacity"))
        self.alpha_slider = gui.Slider(gui.Slider.DOUBLE)
        self.alpha_slider.set_limits(0.05, 1.0)
        self.alpha_slider.double_value = self.alpha
        self.alpha_slider.set_on_value_changed(self._on_alpha_changed)
        self.panel.add_child(self.alpha_slider)

        # --- Scene widget
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1, 1, 1, 1])  # white background
        self.window.add_child(self.scene)

        self.window.set_on_layout(self._on_layout)

        # Add geometry
        self._add_or_update_mesh()
        self._add_or_update_heat()

        # Fit camera
        self._fit_camera()

    def _load_data(self):
        # Mesh
        if self.mesh_path.exists():
            self.mesh = o3d.io.read_triangle_mesh(str(self.mesh_path))
            if not self.mesh.has_vertex_normals():
                self.mesh.compute_vertex_normals()

        # Volume
        self.V, self.origin, self.voxel_size = load_volume_npz(self.volume_path)
        vals = self.V[np.isfinite(self.V)]
        self.vmin = float(np.min(vals))
        self.vmax = float(np.max(vals))
        # start threshold near top-ish so it’s not insanely dense
        self.default_thresh = self.vmin + 0.75 * (self.vmax - self.vmin)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = int(320)
        self.panel.frame = gui.Rect(r.x, r.y, panel_width, r.height)
        self.scene.frame = gui.Rect(r.x + panel_width, r.y, r.width - panel_width, r.height)

    def _fit_camera(self):
        bbox = None
        if self.mesh is not None:
            bbox = self.mesh.get_axis_aligned_bounding_box()
        else:
            # heat-only bbox estimate
            bbox = o3d.geometry.AxisAlignedBoundingBox(self.origin, self.origin + np.array(self.V.shape) * self.voxel_size)

        self.scene.setup_camera(60.0, bbox, bbox.get_center())

    def _add_or_update_mesh(self):
        self.scene.scene.remove_geometry("mesh")
        if self.mesh is None or not self.mesh_visible:
            return

        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        self.scene.scene.add_geometry("mesh", self.mesh, mat)

    def _add_or_update_heat(self):
        self.scene.scene.remove_geometry("heat")

        if not self.heat_visible:
            return

        pcd = make_heatmap_pointcloud(
            self.V, self.origin, self.voxel_size,
            thresh=float(self.thresh_slider.double_value) if hasattr(self, "thresh_slider") else self.default_thresh
        )

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = [1, 1, 1, float(self.alpha)]
        mat.point_size = 3.0  # adjust if needed
        self.scene.scene.add_geometry("heat", pcd, mat)

    def _on_toggle_mesh(self, checked):
        self.mesh_visible = bool(checked)
        self._add_or_update_mesh()

    def _on_toggle_heat(self, checked):
        self.heat_visible = bool(checked)
        self._add_or_update_heat()

    def _on_thresh_changed(self, value):
        self._add_or_update_heat()

    def _on_alpha_changed(self, value):
        self.alpha = float(value)
        # easiest: rebuild heat with new alpha
        self._add_or_update_heat()

    def run(self):
        gui.Application.instance.run()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Run directory (the folder that contains raw/ processed/ exports/)")
    p.add_argument("--mesh", default="processed/open3d_mesh.ply")
    p.add_argument("--volume", default="exports/volume.npz")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run)

    mesh_path = run_dir / args.mesh
    vol_path = run_dir / args.volume

    if not vol_path.exists():
        raise SystemExit(f"ERROR: volume not found: {vol_path}")
    if not mesh_path.exists():
        print(f"WARNING: mesh not found: {mesh_path} (heatmap-only mode)")

    app = App(mesh_path, vol_path)
    app.run()


if __name__ == "__main__":
    main()
