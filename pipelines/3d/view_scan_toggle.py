#!/usr/bin/env python3
"""
view_scan_toggle.py

Interactive FluxSpace viewer: surface mesh + optional magnetic heatmap.

Loads from a canonical run folder (in preference order):
  processed/open3d_mesh_clean.ply  (cleaned mesh, preferred)
  processed/open3d_mesh_raw.ply    (raw TSDF mesh, fallback)
  processed/open3d_mesh.ply        (legacy name, fallback)
  processed/open3d_pcd_clean.ply   (cleaned point cloud, fallback)
  processed/open3d_pcd_raw.ply     (raw point cloud, last resort)
  exports/volume.npz               (voxel heatmap — optional)

If volume.npz is missing the viewer opens in geometry-only mode
(no heatmap controls). This supports camera-only runs.

Controls:
  - Checkbox "Show surface mesh"
  - Checkbox "Show heatmap"  (hidden when no volume loaded)
  - Iso threshold slider
  - Opacity slider
  - "Reframe" button (auto-frames visible geometry)

Heatmap rendering:
  - Option B (default): isosurface via marching cubes (requires scikit-image)
  - Option A (fallback): point cloud from volume voxels above threshold
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_paths import resolve_run_dir  # noqa: E402

try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
except ImportError:
    print("ERROR: open3d is required. Install with: pip install open3d", file=sys.stderr)
    sys.exit(1)

try:
    from skimage import measure as _measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="FluxSpace Viewer (mesh + heatmap)")
    p.add_argument("--run", default="", help="RUN_DIR (or set $RUN_DIR)")
    p.add_argument("--mesh", default="", help="Override path to mesh .ply")
    p.add_argument("--volume", default="", help="Override path to volume .npz")
    p.add_argument("--title", default="FluxSpace Viewer")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_mesh(mesh_path: Path) -> o3d.geometry.TriangleMesh | None:
    if not mesh_path.exists():
        print(f"  Mesh file not found: {mesh_path}")
        return None
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        print(f"  Mesh file is empty: {mesh_path}")
        return None
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    print(f"  Loaded mesh: {mesh_path}  ({len(mesh.vertices)} vertices)")
    return mesh


def _load_point_cloud(pcd_path: Path) -> o3d.geometry.PointCloud | None:
    if not pcd_path.exists():
        return None
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd.is_empty():
        return None
    print(f"  Loaded point cloud: {pcd_path}  ({len(pcd.points)} points)")
    return pcd


def _load_volume_npz(npz_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(f"Volume npz not found: {npz_path}")
    data = np.load(str(npz_path), allow_pickle=True)
    if "volume" in data:
        vol = data["volume"]
    elif "grid" in data:
        vol = data["grid"]
    else:
        raise KeyError(f"{npz_path} missing 'volume' key. Keys: {list(data.keys())}")

    origin = data["origin"].astype(np.float32) if "origin" in data else np.zeros(3, dtype=np.float32)
    voxel_size = float(data["voxel_size"]) if "voxel_size" in data else 0.02
    print(f"  Loaded volume: {npz_path}  shape={vol.shape}  voxel_size={voxel_size}")
    return vol.astype(np.float32), origin, voxel_size


# ---------------------------------------------------------------------------
# Heatmap geometry builders
# ---------------------------------------------------------------------------
def _colormap_value(t: float) -> list[float]:
    """Blue -> cyan -> green -> yellow for t in [0, 1]."""
    if t < 0.33:
        a = t / 0.33
        return [0.0, a, 1.0]
    elif t < 0.66:
        a = (t - 0.33) / 0.33
        return [0.0, 1.0, 1.0 - a]
    else:
        a = (t - 0.66) / 0.34
        return [a, 1.0, 0.0]


def _marching_cubes_mesh(
    volume: np.ndarray, origin: np.ndarray, voxel_size: float, iso: float
) -> o3d.geometry.TriangleMesh | None:
    if not HAS_SKIMAGE:
        return None
    try:
        verts, faces, norms, vals = _measure.marching_cubes(
            volume, level=float(iso),
            spacing=(voxel_size, voxel_size, voxel_size),
        )
    except (ValueError, RuntimeError):
        return None

    if len(verts) == 0:
        return None

    verts_world = verts + origin[None, :]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_normals = o3d.utility.Vector3dVector(norms.astype(np.float64))
    mesh.compute_triangle_normals()

    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax - vmin < 1e-9:
        t = np.zeros_like(vals)
    else:
        t = (vals - vmin) / (vmax - vmin)
    colors = np.array([_colormap_value(u) for u in t], dtype=np.float64)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


def _threshold_point_cloud(
    volume: np.ndarray, origin: np.ndarray, voxel_size: float, iso: float
) -> o3d.geometry.PointCloud | None:
    mask = volume >= iso
    if not np.any(mask):
        return None
    indices = np.argwhere(mask)
    pts = origin[None, :] + indices.astype(np.float64) * voxel_size
    vals = volume[mask]
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax - vmin < 1e-9:
        t = np.zeros(len(vals))
    else:
        t = (vals - vmin) / (vmax - vmin)
    colors = np.array([_colormap_value(u) for u in t], dtype=np.float64)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------
def _geom_bbox(geom) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        bb = geom.get_axis_aligned_bounding_box()
        lo = np.asarray(bb.get_min_bound())
        hi = np.asarray(bb.get_max_bound())
        if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
            return lo, hi
    except Exception:
        pass
    return None


def _union_bbox(*geoms) -> o3d.geometry.AxisAlignedBoundingBox | None:
    lo_all, hi_all = [], []
    for g in geoms:
        if g is None:
            continue
        pair = _geom_bbox(g)
        if pair is not None:
            lo_all.append(pair[0])
            hi_all.append(pair[1])
    if not lo_all:
        return None
    lo = np.min(np.vstack(lo_all), axis=0)
    hi = np.max(np.vstack(hi_all), axis=0)
    return o3d.geometry.AxisAlignedBoundingBox(lo, hi)


# ---------------------------------------------------------------------------
# Viewer class
# ---------------------------------------------------------------------------
class FluxSpaceViewer:
    """Viewer that works with geometry only (no volume) or geometry + heatmap."""

    def __init__(
        self, title: str, mesh: o3d.geometry.TriangleMesh | None,
        volume: np.ndarray | None, origin: np.ndarray | None,
        voxel_size: float,
        surface_pcd: o3d.geometry.PointCloud | None = None,
    ):
        self.mesh = mesh
        self.surface_pcd = surface_pcd
        self.volume = volume
        self.origin = origin if origin is not None else np.zeros(3, dtype=np.float32)
        self.voxel_size = voxel_size
        self.has_volume = volume is not None

        if self.has_volume:
            self.iso = float(np.percentile(self.volume, 95))
        else:
            self.iso = 0.0
        self.opacity = 0.35
        self._heat_geom = None

        # --- Window ---
        self.window = gui.Application.instance.create_window(title, 1280, 780)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([1, 1, 1, 1])

        # --- Panel ---
        self.panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
        self.panel.add_child(gui.Label("FluxSpace Controls"))

        has_surface = mesh is not None or surface_pcd is not None
        label = "Show surface mesh" if mesh is not None else "Show surface (point cloud)"
        self.chk_mesh = gui.Checkbox(label)
        self.chk_mesh.checked = has_surface
        self.chk_mesh.set_on_checked(self._on_toggle_mesh)
        self.panel.add_child(self.chk_mesh)

        # Heatmap controls — only if volume is loaded
        self.chk_heat = None
        self.sld_iso = None
        self.sld_op = None
        if self.has_volume:
            self.chk_heat = gui.Checkbox("Show heatmap")
            self.chk_heat.checked = True
            self.chk_heat.set_on_checked(self._on_toggle_heat)
            self.panel.add_child(self.chk_heat)

            self.panel.add_child(gui.Label("Iso threshold"))
            self.sld_iso = gui.Slider(gui.Slider.DOUBLE)
            vmin = float(np.min(self.volume))
            vmax = float(np.max(self.volume))
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

            mode = "isosurface (marching cubes)" if HAS_SKIMAGE else "point cloud (fallback)"
            self.panel.add_child(gui.Label(f"Heatmap: {mode}"))
        else:
            self.panel.add_child(gui.Label("(Geometry-only mode — no heatmap)"))

        # Reframe button
        self.btn_reframe = gui.Button("Reframe")
        self.btn_reframe.set_on_clicked(self._on_reframe)
        self.panel.add_child(self.btn_reframe)

        # --- Layout ---
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

        # --- Materials ---
        self.mesh_mat = rendering.MaterialRecord()
        self.mesh_mat.shader = "defaultLit"

        self.surface_pcd_mat = rendering.MaterialRecord()
        self.surface_pcd_mat.shader = "defaultUnlit"
        self.surface_pcd_mat.point_size = 2.0

        self.heat_mat = rendering.MaterialRecord()
        self.heat_mat.shader = "defaultLitTransparency"
        self.heat_mat.base_color = [1.0, 1.0, 1.0, self.opacity]

        self.pc_mat = rendering.MaterialRecord()
        self.pc_mat.shader = "defaultUnlit"
        self.pc_mat.point_size = 4.0

        self._build_scene()

    # --- Layout callback ---
    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_w = 300
        self.panel.frame = gui.Rect(r.x, r.y, panel_w, r.height)
        self.scene_widget.frame = gui.Rect(r.x + panel_w, r.y, r.width - panel_w, r.height)

    # --- Scene management ---
    def _safe_remove(self, name: str):
        try:
            self.scene_widget.scene.remove_geometry(name)
        except Exception:
            pass

    def _add_surface(self):
        if self.mesh is not None:
            self.scene_widget.scene.add_geometry("surface", self.mesh, self.mesh_mat)
        elif self.surface_pcd is not None:
            self.scene_widget.scene.add_geometry("surface", self.surface_pcd, self.surface_pcd_mat)

    def _build_scene(self):
        self.scene_widget.scene.clear_geometry()
        if self.chk_mesh.checked:
            self._add_surface()
        if self.has_volume:
            self._rebuild_heat()
        self._on_reframe()

    def _rebuild_heat(self):
        self._safe_remove("heat")
        self._heat_geom = None

        if not self.has_volume:
            return
        if self.chk_heat is not None and not self.chk_heat.checked:
            return

        iso = float(self.sld_iso.double_value) if self.sld_iso else self.iso

        geom = None
        mat = self.heat_mat
        if HAS_SKIMAGE:
            mc = _marching_cubes_mesh(self.volume, self.origin, self.voxel_size, iso)
            if mc is not None and not mc.is_empty():
                geom = mc
                opacity = float(self.sld_op.double_value) if self.sld_op else self.opacity
                self.heat_mat.base_color = [1.0, 1.0, 1.0, opacity]
                mat = self.heat_mat

        if geom is None:
            pc = _threshold_point_cloud(self.volume, self.origin, self.voxel_size, iso)
            if pc is not None and not pc.is_empty():
                geom = pc
                mat = self.pc_mat

        if geom is not None:
            self.scene_widget.scene.add_geometry("heat", geom, mat)
            self._heat_geom = geom

    # --- Callbacks ---
    def _on_toggle_mesh(self, checked: bool):
        self._safe_remove("surface")
        if checked:
            self._add_surface()

    def _on_toggle_heat(self, checked: bool):
        self._safe_remove("heat")
        self._heat_geom = None
        if checked:
            self._rebuild_heat()

    def _on_iso_changed(self, _):
        if self.chk_heat is not None and self.chk_heat.checked:
            self._rebuild_heat()

    def _on_opacity_changed(self, _):
        if self.chk_heat is not None and self.chk_heat.checked:
            self._rebuild_heat()

    def _on_reframe(self):
        geoms = []
        if self.chk_mesh.checked:
            if self.mesh is not None:
                geoms.append(self.mesh)
            elif self.surface_pcd is not None:
                geoms.append(self.surface_pcd)
        if self._heat_geom is not None:
            geoms.append(self._heat_geom)

        if not geoms:
            if self.mesh is not None:
                geoms.append(self.mesh)
            elif self.surface_pcd is not None:
                geoms.append(self.surface_pcd)
            if self.has_volume:
                pc_any = _threshold_point_cloud(
                    self.volume, self.origin, self.voxel_size, float(np.min(self.volume))
                )
                if pc_any is not None:
                    geoms.append(pc_any)

        if not geoms:
            return

        bb = _union_bbox(*geoms)
        if bb is None:
            return

        center = np.asarray(bb.get_center(), dtype=np.float64)
        extent = np.asarray(bb.get_extent(), dtype=np.float64)
        radius = float(np.linalg.norm(extent)) * 0.6 + 1e-6
        eye = center + np.array([radius, radius, radius], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        try:
            self.scene_widget.look_at(center, eye, up)
        except AttributeError:
            try:
                self.scene_widget.scene.setup_camera(60.0, bb, center)
            except Exception:
                pass
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    # --- Resolve run directory ---
    try:
        run = resolve_run_dir(args.run if args.run else None)
    except ValueError:
        if not args.mesh:
            print("ERROR: Provide --run (or $RUN_DIR), or --mesh.", file=sys.stderr)
            return 2
        run = None

    # --- Mesh path (prefer clean over raw) ---
    mesh = None
    surface_pcd = None

    if args.mesh:
        mesh = _load_mesh(Path(args.mesh).expanduser().resolve())
    elif run:
        mesh_candidates = [
            run / "processed" / "open3d_mesh_clean.ply",
            run / "processed" / "open3d_mesh_raw.ply",
            run / "processed" / "open3d_mesh.ply",  # legacy name
        ]
        for mp in mesh_candidates:
            mesh = _load_mesh(mp)
            if mesh is not None:
                print(f"  -> Using mesh: {mp.name}")
                break

    # Point cloud fallback
    if mesh is None and run:
        pcd_candidates = [
            run / "processed" / "open3d_pcd_clean.ply",
            run / "processed" / "open3d_pcd_raw.ply",
        ]
        for pp in pcd_candidates:
            surface_pcd = _load_point_cloud(pp)
            if surface_pcd is not None:
                print(f"  -> Using surface pcd: {pp.name}")
                break

    # --- Volume path (optional) ---
    volume = None
    origin = None
    voxel_size = 0.02

    if args.volume:
        vol_path = Path(args.volume).expanduser().resolve()
    elif run:
        candidates = [
            run / "exports" / "volume.npz",
            run / "exports" / "volume_fixed.npz",
        ]
        vol_path = next((p for p in candidates if p.exists()), None)
    else:
        vol_path = None

    if vol_path is not None and vol_path.exists():
        try:
            volume, origin, voxel_size = _load_volume_npz(vol_path)
        except Exception as exc:
            print(f"WARNING: Could not load volume ({exc}); continuing in geometry-only mode.")
            traceback.print_exc()
    elif args.volume:
        print(f"WARNING: Volume file not found: {args.volume}; continuing in geometry-only mode.")

    # --- Determine mode ---
    if mesh is None and surface_pcd is None and volume is None:
        print("ERROR: No geometry and no volume found. Nothing to display.", file=sys.stderr)
        return 2

    if volume is None:
        if mesh is None and surface_pcd is None:
            print("ERROR: No geometry found and no volume. Nothing to display.", file=sys.stderr)
            return 2
        print("  Geometry-only mode (no heatmap volume loaded)")
    elif mesh is None and surface_pcd is None:
        print("  Heatmap-only mode (no geometry loaded)")
    else:
        if mesh is None:
            print("  No mesh — using point cloud for surface")
        print("  Full mode (geometry + heatmap)")

    print("  Starting viewer...")

    gui.Application.instance.initialize()
    FluxSpaceViewer(args.title, mesh, volume, origin, voxel_size, surface_pcd=surface_pcd)
    gui.Application.instance.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
