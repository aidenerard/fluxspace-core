# Explanation of `view_scan_toggle.py`

This document explains the interactive 3D viewer that displays reconstructed geometry and optional magnetic heatmap overlays from a FluxSpace run directory.

---

## Overview

**`view_scan_toggle.py`** opens an Open3D GUI window showing:
- **Surface geometry** — a mesh (preferred) or point cloud from the reconstruction step.
- **Magnetic heatmap** (optional) — an isosurface or thresholded point cloud rendered from `volume.npz`.

The viewer supports **geometry-only mode** (no heatmap controls) for camera-only runs where no magnetic volume exists.

---

## What it does

1. **Resolve run directory** from `--run` or `$RUN_DIR`.
2. **Load geometry** in preference order:
   - `processed/open3d_mesh_clean.ply` (cleaned mesh — preferred)
   - `processed/open3d_mesh_raw.ply` (raw TSDF mesh)
   - `processed/open3d_mesh.ply` (legacy name)
   - `processed/open3d_pcd_clean.ply` (cleaned point cloud — fallback)
   - `processed/open3d_pcd_raw.ply` (raw point cloud — last resort)
3. **Load volume** (optional):
   - `exports/volume.npz` or `exports/volume_fixed.npz`
   - If the volume cannot be loaded or doesn't exist, the viewer proceeds in **geometry-only mode**.
4. **Determine display mode:**
   - **Full mode:** geometry + heatmap controls (volume loaded).
   - **Geometry-only mode:** geometry only (no heatmap controls).
   - **Heatmap-only mode:** volume only (no geometry found).
5. **Create GUI window** with a side panel containing controls and a 3D scene.
6. **Auto-frame** the camera to show all loaded geometry.

---

## GUI controls

### Always present

| Control | Description |
|---------|-------------|
| **Show surface mesh** (checkbox) | Toggle the surface geometry on/off. Label changes to "Show surface (point cloud)" if only a point cloud is loaded. |
| **Reframe** (button) | Reset the camera to frame all visible geometry. Useful after toggling layers or if the view gets lost. |

### Heatmap controls (only when volume is loaded)

| Control | Description |
|---------|-------------|
| **Show heatmap** (checkbox) | Toggle the heatmap overlay on/off. |
| **Iso threshold** (slider) | Controls the isosurface level. Only voxels at or above this value are shown. Default: 95th percentile of the volume. |
| **Heat opacity** (slider) | Transparency of the heatmap surface (0.05 = nearly transparent, 1.0 = opaque). |

When no volume is loaded, the panel shows "(Geometry-only mode — no heatmap)" instead.

---

## Heatmap rendering

Two rendering methods are available, selected automatically:

### Option B (default): Isosurface via marching cubes

Uses `skimage.measure.marching_cubes` to extract a smooth isosurface from the volume at the current threshold. Produces a coloured triangle mesh with vertex colours mapped from the volume values (blue → cyan → green → yellow).

### Option A (fallback): Thresholded point cloud

Used when `scikit-image` is not installed. Selects all voxels above the threshold and renders them as a coloured point cloud. Simpler but less visually smooth.

---

## Bounding box handling

The viewer implements manual bounding box union to avoid Open3D version incompatibilities. The `_union_bbox()` helper computes the enclosing box of multiple geometries by stacking min/max corners, rather than using the `+` operator on `AxisAlignedBoundingBox` (which is unsupported in some Open3D versions).

Camera framing uses `scene_widget.look_at()` with a fallback to `scene.setup_camera()` for older Open3D versions.

---

## CLI arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--run DIR` | No | Run directory path (or set `$RUN_DIR`). |
| `--mesh PATH` | No | Override path to a specific mesh `.ply` file. |
| `--volume PATH` | No | Override path to a specific volume `.npz` file. |
| `--title TEXT` | No | Window title (default: "FluxSpace Viewer"). |

At least `--run` or `--mesh` must be provided.

---

## Example usage

```bash
# Standard usage (from run directory)
python3 pipelines/3d/view_scan_toggle.py --run data/runs/run_20260210_1430

# Camera-only run (no volume — opens in geometry-only mode)
python3 pipelines/3d/view_scan_toggle.py --run data/runs/run_cam_only_20260210

# Explicit file paths
python3 pipelines/3d/view_scan_toggle.py \
  --mesh processed/open3d_mesh_clean.ply \
  --volume exports/volume.npz

# Custom title (used by run_all_3d.sh)
python3 pipelines/3d/view_scan_toggle.py \
  --run data/runs/run_20260210_1430 \
  --title "FluxSpace — run_20260210_1430"
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `open3d` | 3D rendering, GUI, mesh/point cloud I/O. |
| `numpy` | Volume array operations, bounding box math. |
| `scikit-image` (optional) | Marching cubes for isosurface rendering. Falls back to point cloud without it. |

---

## Key classes

### `FluxSpaceViewer`

The main viewer class. Accepts optional `volume` (can be `None` for geometry-only mode). Manages:
- Scene geometry (surface + heatmap layers)
- Material records for lit/unlit/transparent rendering
- GUI panel with conditional heatmap controls
- Camera framing with cross-version Open3D compatibility

---

## Relation to other 3D scripts

- **Before:** `open3d_reconstruct.py` and `clean_geometry.py` produce the geometry files. `mag_world_to_voxel_volume.py` produces `volume.npz`.
- **Pipeline runner:** Called as the final step by `run_all_3d.sh` (unless `--no-viewer`).
- **Standalone:** Can be re-opened at any time on any completed run without re-running the pipeline.

See [PIPELINE_3D.md](PIPELINE_3D.md) for the full 3D runbook.
