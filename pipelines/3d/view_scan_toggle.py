#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="RUN_DIR (e.g. data/runs/run_YYYYMMDD_HHMM)")
    p.add_argument("--mesh", default="", help="Optional override path to mesh .ply")
    p.add_argument("--volume", default="", help="Optional override path to volume .npz")
    p.add_argument("--title", default="FluxSpace Viewer (Mesh + Heatmap)")
    return p.parse_args()


def _load_mesh(mesh_path: Path) -> o3d.geometry.TriangleMesh | None:
    if not mesh_path.exists():
        return None
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        return None
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


def _load_volume_npz(npz_path: Path):
    """
    Expected keys (your mag_world_to_voxel_volume.py output):
      - volume: (nx, ny, nz) float array
      - origin: (3,) float, world origin of voxel (meters)
      - voxel_size: float (meters)
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"volume npz not found: {npz_path}")
    data = np.load(str(npz_path), allow_pickle=True)
    # common variants
    if "volume" in data:
        vol = data["volume"]
    elif "grid" in data:
        vol = data["grid"]
    else:
        raise KeyError(f"{npz_path} missing 'volume' (or 'grid') key. Keys: {list(data.keys())}")

    origin = data["origin"] if "origin" in data else np.array([0.0, 0.0, 0.0], dtype=float)
    voxel_size = float(data["voxel_size"]) if "voxel_size" in data else 0.02
    return vol.astype(np.float32), origin.astype(np.float32), voxel_size


def _marching_cubes_mesh(volume: np.ndarray, origin: np.ndarray, voxel_size: float, iso: float):
    """
    Returns an Open3D TriangleMesh from marching cubes isosurface at 'iso'.
    Uses scikit-image if available.
    """
    try:
        from skimage import measure
    except Exception as e:
        raise RuntimeError(
            "Option B requires scikit-image for marching cubes.\n"
            "Install: pip install scikit-image\n"
            f"Import error: {e}"
        )

    # marching_cubes expects (z, y, x) or any order; it treats axes consistently.
    # We'll keep volume as (nx, ny, nz) from our pipeline and pass spacing=(voxel_size, voxel_size, voxel_size)
    # It returns vertices in voxel coordinates scaled by spacing.
    verts, faces, norms, vals = measure.marching_cubes(volume, level=float(iso), spacing=(voxel_size, voxel_size, voxel_size))

    # verts currently in (x, y, z) in world units relative to (0,0,0) of grid.
    verts_world = verts + origin[None, :]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_normals = o3d.utility.Vector3dVector(norms.astype(np.float64))
    mesh.compute_triangle_normals()

    # Color the isosurface by the sampled scalar field (vals)
    vmin = float(np.min(vals)) if len(vals) else 0.0
    vmax = float(np.max(vals)) if len(vals) else 1.0
    if vmax - vmin < 1e-9:
        t = np.zeros_like(vals, dtype=np.float32)
    else:
        t = (vals.astype(np.float32) - vmin) / (vmax - vmin)

    # simple colormap: blue->cyan->green->yellow (no extra deps)
    # (looks "heatmap-ish" without matplotlib)
    colors = np.zeros((len(t), 3), dtype=np.float32)
    # piecewise
    for i, u in enumerate(t):
        if u < 0.33:
            a = u / 0.33
            colors[i] = [0.0, a, 1.0]          # blue -> cyan
        elif u < 0.66:
            a = (u - 0.33) / 0.33
            colors[i] = [0.0, 1.0, 1.0 - a]    # cyan -> green
        else:
            a = (u - 0.66) / 0.34
            colors[i] = [a, 1.0, 0.0]          # green -> yellow
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return mesh


class FluxSpaceViewer:
    def __init__(self, title: str, mesh: o3d.geometry.TriangleMesh | None,
                 volume: np.ndarray, origin: np.ndarray, voxel_size: float):
        self.mesh = mesh
        self.volume = volume
        self.origin = origin
        self.voxel_size = voxel_size

        self.iso = float(np.percentile(self.volume, 95))  # good starting point
        self.opacity = 0.35

        self.window = gui.Application.instance.create_window(title, 1280, 780)

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([1, 1, 1, 1])

        self.panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
        self.panel.add_child(gui.Label(""))

        self.chk_mesh = gui.Checkbox("Show surface mesh")
        self.chk_mesh.checked = True
        self.chk_mesh.set_on_checked(self._on_toggle_mesh)

        self.chk_heat = gui.Checkbox("Show heatmap (isosurface)")
        self.chk_heat.checked = True
        self.chk_heat.set_on_checked(self._on_toggle_heat)

        self.panel.add_child(self.chk_mesh)
        self.panel.add_child(self.chk_heat)

        self.panel.add_child(gui.Label("Iso threshold (surface at value >= iso)"))
        self.sld_iso = gui.Slider(gui.Slider.DOUBLE)
        vmin = float(np.min(self.volume))
        vmax = float(np.max(self.volume))
        # Clamp in case of weird data
        if vmax - vmin < 1e-9:
            vmax = vmin + 1.0
        self.sld_iso.set_limits(vmin, vmax)
        self.sld_iso.double_value = self.iso
        self.sld_iso.set_on_value_changed(self._on_iso_changed)
        self.panel.add_child(self.sld_iso)

        self.panel.add_child(gui.Label("Heat opacity"))
        self.sld_op = gui.Slider(gui.Slider.DOUBLE)
        self.sld_op.set_limits(0.05, 1.0)
        self.sld_op.double_value = self.opacity
        self.sld_op.set_on_value_changed(self._on_opacity_changed)
        self.panel.add_child(self.sld_op)

        self.btn_reframe = gui.Button("Reframe")
        self.btn_reframe.set_on_clicked(self._on_reframe)
        self.panel.add_child(self.btn_reframe)

        # Layout
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

        # Materials
        self.mesh_mat = rendering.MaterialRecord()
        self.mesh_mat.shader = "defaultLit"

        self.heat_mat = rendering.MaterialRecord()
        self.heat_mat.shader = "defaultLitTransparency"
        self.heat_mat.base_color = [1.0, 1.0, 1.0, self.opacity]  # alpha controls opacity

        self._heat_geom = None

        self._build_scene()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_w = 300
        self.panel.frame = gui.Rect(r.x, r.y, panel_w, r.height)
        self.scene_widget.frame = gui.Rect(r.x + panel_w, r.y, r.width - panel_w, r.height)

    def _clear_scene(self):
        self.scene_widget.scene.clear_geometry()

    def _build_scene(self):
        self._clear_scene()

        if self.mesh is not None:
            self.scene_widget.scene.add_geometry("surface", self.mesh, self.mesh_mat)

        self._rebuild_heat()

        self._on_reframe()

    def _rebuild_heat(self):
        # Remove old heat geometry if present
        try:
            self.scene_widget.scene.remove_geometry("heat")
        except Exception:
            pass

        if not self.chk_heat.checked:
            return

        iso = float(self.sld_iso.double_value)

        heat_mesh = _marching_cubes_mesh(self.volume, self.origin, self.voxel_size, iso)
        if heat_mesh.is_empty():
            return

        # update opacity
        self.heat_mat.base_color = [1.0, 1.0, 1.0, float(self.sld_op.double_value)]
        self.scene_widget.scene.add_geometry("heat", heat_mesh, self.heat_mat)

        self._heat_geom = heat_mesh

    def _on_toggle_mesh(self, checked: bool):
        try:
            self.scene_widget.scene.remove_geometry("surface")
        except Exception:
            pass
        if checked and self.mesh is not None:
            self.scene_widget.scene.add_geometry("surface", self.mesh, self.mesh_mat)

    def _on_toggle_heat(self, checked: bool):
        try:
            self.scene_widget.scene.remove_geometry("heat")
        except Exception:
            pass
        if checked:
            self._rebuild_heat()

    def _on_iso_changed(self, _):
        # rebuilding on every slider tick can be heavy; but your volume is small (like 67x38x38) so it's fine.
        if self.chk_heat.checked:
            self._rebuild_heat()

    def _on_opacity_changed(self, _):
        # only need to update material; easiest: rebuild heat geometry with new alpha
        if self.chk_heat.checked:
            self._rebuild_heat()

    def _combined_bounds(self):
        boxes = []
        if self.chk_mesh.checked and self.mesh is not None and not self.mesh.is_empty():
            boxes.append(self.mesh.get_axis_aligned_bounding_box())
        if self.chk_heat.checked and self._heat_geom is not None and not self._heat_geom.is_empty():
            boxes.append(self._heat_geom.get_axis_aligned_bounding_box())

        if not boxes:
            return None

        mins = np.vstack([b.get_min_bound() for b in boxes])
        maxs = np.vstack([b.get_max_bound() for b in boxes])

        bb = o3d.geometry.AxisAlignedBoundingBox(
            mins.min(axis=0),
            maxs.max(axis=0),
        )
        return bb


    def _on_reframe(self):
        bb = self._combined_bounds()
        if bb is None:
            return

        center = bb.get_center()
        extent = bb.get_extent()
        radius = float(np.linalg.norm(extent)) * 0.6 + 1e-6

        # Robust across Open3D versions:
        # Use look_at (exists broadly) instead of scene.setup_camera (missing on your build)
        eye = center + np.array([radius, radius, radius], dtype=float)
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        try:
            self.scene_widget.look_at(center, eye, up)
        except Exception:
            # last resort: ignore if old build
            pass


def main():
    args = parse_args()
    run = Path(args.run).expanduser().resolve()

    # mesh
    if args.mesh:
        mesh_path = Path(args.mesh).expanduser().resolve()
    else:
        # try a few common names
        cand = [
            run / "processed" / "open3d_mesh_clean.ply",
            run / "processed" / "open3d_mesh.ply",
            run / "raw" / "oak_rgbd" / "open3d_mesh.ply",
        ]
        mesh_path = next((p for p in cand if p.exists()), cand[1])

    mesh = _load_mesh(mesh_path)

    # volume
    if args.volume:
        vol_path = Path(args.volume).expanduser().resolve()
    else:
        cand = [
            run / "exports" / "volume.npz",
            run / "exports" / "volume_fixed.npz",
        ]
        vol_path = next((p for p in cand if p.exists()), cand[0])

    volume, origin, voxel_size = _load_volume_npz(vol_path)

    gui.Application.instance.initialize()
    FluxSpaceViewer(args.title, mesh, volume, origin, voxel_size)
    gui.Application.instance.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
