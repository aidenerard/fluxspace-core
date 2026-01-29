# Smoke Tests

Quick checks that entrypoints and pipelines work. Run from **repo root** (`~/fluxspace-core` or your clone).

---

## 2D entrypoints (help only)

```bash
cd ~/fluxspace-core

python3 pipelines/2d/mag_to_csv.py --help
python3 pipelines/2d/validate_and_diagnosticsV1.py --help
python3 pipelines/2d/compute_local_anomaly_v2.py --help
python3 pipelines/2d/interpolate_to_heatmapV1.py --help
python3 pipelines/2d/interpolate_to_Btotal_heatmap.py --help
```

Expected: each prints argparse help. No sensor or data required.

---

## 3D entrypoints (help only)

```bash
python3 pipelines/3d/fuse_mag_with_trajectory.py --help
python3 pipelines/3d/polycam_raw_to_trajectory.py --help
python3 pipelines/3d/rtabmap_poses_to_trajectory.py --help
python3 pipelines/3d/mag_world_to_voxel_volume.py --help
python3 pipelines/3d/visualize_3d_heatmap.py --help
```

Expected: each prints help. No USB/Polycam/RTAB-Map required.

---

## 2D run script (no data change)

```bash
./tools/2d/new_run.sh
```

Expected: creates `data/runs/<MM-DD-YYYY_HH-MM>/` with `raw/`, `processed/`, `exports/` and copies current `data/raw`, `data/processed`, `data/exports` into it. Safe to run; only creates a new folder and copies.

---

## 3D scan script (no data change)

```bash
./tools/3d/new_3d_scan.sh
./tools/3d/new_3d_scan.sh --label smoke
```

Expected: creates `data/scans/<MM-DD-YYYY_HH-MM>__3d/` and optionally `...__3d__smoke/` with `raw/`, `processed/`, `exports/`, `config/`. Safe to run.

---

## Wrapper → implementation

Running a pipeline script in `pipelines/2d/` or `pipelines/3d/` should execute the corresponding file in `pipelines/2d/` or `pipelines/3d/`. For example:

```bash
python3 pipelines/2d/mag_to_csv.py --help
```

shows the same help as the implementation in `pipelines/2d/mag_to_csv.py`. No code changes needed for existing commands.
