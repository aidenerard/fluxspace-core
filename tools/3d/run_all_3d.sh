#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_all_3d.sh — One-command 3D pipeline runner for FluxSpace
#
# Runs:  reconstruct -> clean -> [mag fusion -> voxelise] -> viewer
# Works on Mac and Pi.  No manual RUN_DIR export needed.
#
# Usage examples:
#   ./tools/3d/run_all_3d.sh --new                          # fresh run
#   ./tools/3d/run_all_3d.sh --latest                       # re-process newest run
#   ./tools/3d/run_all_3d.sh --run data/runs/run_20260210_1430
#   ./tools/3d/run_all_3d.sh --latest --camera-only         # geometry only, no mag
#   ./tools/3d/run_all_3d.sh --latest --no-viewer --verbose
#   ./tools/3d/run_all_3d.sh --latest --quality high --every-n 2
# ============================================================================

# ── Colours (disabled when stdout isn't a terminal) ────────────────────────
if [[ -t 1 ]]; then
  _B='\033[1m'; _R='\033[0m'; _G='\033[32m'; _Y='\033[33m'; _RED='\033[31m'
else
  _B=''; _R=''; _G=''; _Y=''; _RED=''
fi

banner()  { printf "\n${_B}═══ %s${_R}\n" "$*"; }
info()    { printf "${_G}✓${_R} %s\n" "$*"; }
warn()    { printf "${_Y}⚠${_R} %s\n" "$*" >&2; }
die()     { printf "${_RED}✗ ERROR:${_R} %s\n" "$*" >&2; exit 1; }

# ── Defaults ───────────────────────────────────────────────────────────────
RUN_DIR=""
MODE=""                # run | latest | new
REPO_ROOT=""
NO_VIEWER=false
VOXEL_SIZE="0.02"
MAX_DIM="256"
MAG_OVERRIDE=""
OAK_OVERRIDE=""
EXT_OVERRIDE=""
DEFAULT_EXTRINSICS=""
VERBOSE=false
QUALITY=""             # fast | balanced | high (empty = balanced)
CLEAN_VOXEL=""         # override for clean_geometry --downsample
CLEAN_SOR_NB=""        # override for clean_geometry --sor-nb-neighbors
CLEAN_SOR_STD=""       # override for clean_geometry --sor-std-ratio
SKIP_CLEAN=false       # skip cleaning step entirely
EVERY_N=""             # frame subsample for reconstruction
MAX_FRAMES=""          # max frames for reconstruction
DEPTH_TRUNC=""         # depth truncation override
RECON_VOXEL=""         # reconstruction voxel size override
ODO_METHOD=""          # odometry method override
SAVE_GLB=false         # export GLB files
CAMERA_ONLY=false      # skip all mag-related steps

# ── Usage ──────────────────────────────────────────────────────────────────
usage() {
  cat <<'EOF'
Usage: run_all_3d.sh <mode> [options]

Modes (pick one):
  --run <RUN_DIR>       Use an existing run directory
  --latest              Use the newest data/runs/run_* directory
  --new                 Create data/runs/run_YYYYMMDD_HHMM

Pipeline mode:
  --camera-only         Geometry only — skip mag fusion / voxelisation / heatmap.
                        Auto-detected if raw/mag_run.csv is missing.

Reconstruction options:
  --every-n <int>       Use every Nth frame (default: 1)
  --max-frames <int>    Stop reconstruction after N frames
  --depth-trunc <float> Max depth in metres (default: 3.0)
  --recon-voxel <float> TSDF voxel size (default: 0.01)
  --odometry <method>   hybrid | color (default: hybrid)
  --save-glb            Export meshes as GLB for web viewing

Cleaning options:
  --quality <preset>    fast | balanced | high  (default: balanced)
  --skip-clean          Skip geometry cleaning step
  --clean-voxel <float> Override clean downsample voxel size
  --clean-sor-nb <int>  Override SOR neighbour count
  --clean-sor-std <f>   Override SOR std ratio

Voxelisation options (ignored in --camera-only):
  --voxel-size <float>  Mag volume voxel edge (default: 0.02)
  --max-dim <int>       Max voxels per axis (default: 256)

Input overrides:
  --mag <path>          Override magnetometer CSV (default: raw/mag_run.csv)
  --oak <path>          Override OAK capture dir  (default: raw/oak_rgbd)
  --extrinsics <path>   Override extrinsics JSON  (default: raw/extrinsics.json)
  --default-extrinsics  Shorthand mount offset, e.g. "behind_cm=2,down_cm=10"

General:
  --repo-root <path>    Repository root (default: auto-detect from git)
  --no-viewer           Skip the Open3D viewer at the end
  --verbose             Print full Python output
  -h, --help            Show this help
EOF
}

# ── Arg parsing ────────────────────────────────────────────────────────────
[[ $# -eq 0 ]] && { usage; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      [[ -z "${2:-}" ]] && die "--run requires a directory path"
      MODE="run"; RUN_DIR="$2"; shift 2 ;;
    --latest)
      MODE="latest"; shift ;;
    --new)
      MODE="new"; shift ;;
    --camera-only)
      CAMERA_ONLY=true; shift ;;
    --repo-root)
      [[ -z "${2:-}" ]] && die "--repo-root requires a path"
      REPO_ROOT="$2"; shift 2 ;;
    --no-viewer)
      NO_VIEWER=true; shift ;;
    --skip-clean|--no-clean)
      SKIP_CLEAN=true; shift ;;
    --quality)
      [[ -z "${2:-}" ]] && die "--quality requires fast|balanced|high"
      QUALITY="$2"; shift 2 ;;
    --every-n)
      [[ -z "${2:-}" ]] && die "--every-n requires a value"
      EVERY_N="$2"; shift 2 ;;
    --max-frames)
      [[ -z "${2:-}" ]] && die "--max-frames requires a value"
      MAX_FRAMES="$2"; shift 2 ;;
    --depth-trunc)
      [[ -z "${2:-}" ]] && die "--depth-trunc requires a value"
      DEPTH_TRUNC="$2"; shift 2 ;;
    --recon-voxel)
      [[ -z "${2:-}" ]] && die "--recon-voxel requires a value"
      RECON_VOXEL="$2"; shift 2 ;;
    --odometry)
      [[ -z "${2:-}" ]] && die "--odometry requires hybrid|color"
      ODO_METHOD="$2"; shift 2 ;;
    --save-glb)
      SAVE_GLB=true; shift ;;
    --clean-voxel)
      [[ -z "${2:-}" ]] && die "--clean-voxel requires a value"
      CLEAN_VOXEL="$2"; shift 2 ;;
    --clean-sor-nb)
      [[ -z "${2:-}" ]] && die "--clean-sor-nb requires a value"
      CLEAN_SOR_NB="$2"; shift 2 ;;
    --clean-sor-std)
      [[ -z "${2:-}" ]] && die "--clean-sor-std requires a value"
      CLEAN_SOR_STD="$2"; shift 2 ;;
    --voxel-size)
      [[ -z "${2:-}" ]] && die "--voxel-size requires a value"
      VOXEL_SIZE="$2"; shift 2 ;;
    --max-dim)
      [[ -z "${2:-}" ]] && die "--max-dim requires a value"
      MAX_DIM="$2"; shift 2 ;;
    --mag)
      [[ -z "${2:-}" ]] && die "--mag requires a path"
      MAG_OVERRIDE="$2"; shift 2 ;;
    --oak)
      [[ -z "${2:-}" ]] && die "--oak requires a path"
      OAK_OVERRIDE="$2"; shift 2 ;;
    --extrinsics)
      [[ -z "${2:-}" ]] && die "--extrinsics requires a path"
      EXT_OVERRIDE="$2"; shift 2 ;;
    --default-extrinsics)
      [[ -z "${2:-}" ]] && die "--default-extrinsics requires a value like 'behind_cm=2,down_cm=10'"
      DEFAULT_EXTRINSICS="$2"; shift 2 ;;
    --verbose)
      VERBOSE=true; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      die "Unknown option: $1   (run with --help)" ;;
  esac
done

[[ -z "$MODE" ]] && die "Must specify one of --run, --latest, or --new."

# ── Repo root ──────────────────────────────────────────────────────────────
if [[ -n "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"

# ── Python check ───────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  die "python3 not found. Activate your venv first:\n  source .venv/bin/activate"
fi

_py_check() {
  python3 -c "import $1" 2>/dev/null && return 0
  warn "Python module '$1' not importable."
  printf "  Hint: activate your venv and install it:\n"
  printf "    source .venv/bin/activate && pip install %s\n" "$2"
  return 1
}

banner "Checking Python dependencies"
_py_check numpy   numpy
_py_check open3d  open3d  || die "open3d is required for reconstruction + viewer."
if ! $CAMERA_ONLY; then
  _py_check pandas  pandas
  _py_check scipy   scipy   || warn "scipy missing — scatter method still works."
fi
info "Core dependencies OK"

# ── Quality presets ───────────────────────────────────────────────────────
case "${QUALITY:-balanced}" in
  fast)
    : "${CLEAN_VOXEL:=0.01}"
    : "${CLEAN_SOR_NB:=20}"
    : "${CLEAN_SOR_STD:=2.5}"
    info "Quality preset: fast" ;;
  balanced|"")
    : "${CLEAN_VOXEL:=0.005}"
    : "${CLEAN_SOR_NB:=30}"
    : "${CLEAN_SOR_STD:=2.0}"
    info "Quality preset: balanced" ;;
  high)
    : "${CLEAN_VOXEL:=0.003}"
    : "${CLEAN_SOR_NB:=40}"
    : "${CLEAN_SOR_STD:=1.8}"
    warn "Quality preset: high (may be slower)" ;;
  *)
    die "Unknown --quality '$QUALITY'. Use fast, balanced, or high." ;;
esac

# ── Run directory selection ────────────────────────────────────────────────
banner "Selecting run directory"

case "$MODE" in
  run)
    RUN_DIR="$(cd "$REPO_ROOT" && python3 -c "from pathlib import Path; print(Path('$RUN_DIR').expanduser().resolve())")"
    [[ -d "$RUN_DIR" ]] || die "Run directory not found: $RUN_DIR"
    ;;
  latest)
    RUNS_BASE="$REPO_ROOT/data/runs"
    [[ -d "$RUNS_BASE" ]] || die "No data/runs/ directory found in $REPO_ROOT"
    RUN_DIR="$(ls -1d "$RUNS_BASE"/run_* 2>/dev/null | sort | tail -n1)" || true
    [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] || die "No run_* directories found under $RUNS_BASE"
    ;;
  new)
    STAMP="$(date +%Y%m%d_%H%M)"
    RUN_DIR="$REPO_ROOT/data/runs/run_${STAMP}"
    mkdir -p "$RUN_DIR"/{raw,processed,exports}
    info "Created new run directory"
    ;;
esac

info "RUN_DIR = $RUN_DIR"

# ── Resolve input paths ───────────────────────────────────────────────────
OAK_DIR="${OAK_OVERRIDE:-$RUN_DIR/raw/oak_rgbd}"
MAG_CSV="${MAG_OVERRIDE:-$RUN_DIR/raw/mag_run.csv}"
EXT_JSON="${EXT_OVERRIDE:-$RUN_DIR/raw/extrinsics.json}"

mkdir -p "$RUN_DIR"/{raw,processed,exports}

# ── Auto-detect camera-only mode ──────────────────────────────────────────
if ! $CAMERA_ONLY && [[ ! -f "$MAG_CSV" ]]; then
  CAMERA_ONLY=true
  warn "mag_run.csv not found — auto-switching to camera-only mode"
fi

if $CAMERA_ONLY; then
  banner "Camera-only mode"
  info "Skipping magnetometer requirements, fusion, and voxelisation steps"
  info "Pipeline: reconstruct -> clean -> viewer (geometry only)"
fi

# ── Validation ─────────────────────────────────────────────────────────────
banner "Validating inputs"

FAIL=false

_require_dir() {
  if [[ ! -d "$1" ]]; then
    printf "${_RED}✗${_R} Missing directory: %s\n" "$1" >&2
    FAIL=true
  else
    info "Found: $1"
  fi
}

_require_file() {
  if [[ ! -f "$1" ]]; then
    printf "${_RED}✗${_R} Missing file: %s\n" "$1" >&2
    FAIL=true
  else
    info "Found: $1"
  fi
}

_require_dir  "$OAK_DIR/color"
_require_dir  "$OAK_DIR/depth"
_require_file "$OAK_DIR/timestamps.csv"

if [[ ! -f "$OAK_DIR/intrinsics.json" ]]; then
  warn "intrinsics.json not found — reconstruction will use approximate values"
else
  info "Found: $OAK_DIR/intrinsics.json"
fi

if ! $CAMERA_ONLY; then
  # Full mode: require mag file
  if [[ ! -f "$MAG_CSV" ]]; then
    printf "${_RED}✗${_R} Missing mag file: %s\n" "$MAG_CSV" >&2
    FAIL=true
  else
    info "Found: $MAG_CSV"
  fi

  if [[ ! -f "$RUN_DIR/raw/calibration.json" ]]; then
    warn "calibration.json not found — not critical, continuing"
  else
    info "Found: $RUN_DIR/raw/calibration.json"
  fi

  if [[ ! -f "$EXT_JSON" ]]; then
    if [[ -n "$DEFAULT_EXTRINSICS" ]]; then
      info "No extrinsics.json — will use --default-extrinsics \"$DEFAULT_EXTRINSICS\""
    else
      warn "extrinsics.json not found — fusion will use identity offset"
    fi
  else
    info "Found: $EXT_JSON"
  fi
fi

$FAIL && die "Required input files missing (see above). Place raw data in $RUN_DIR/raw/ and re-run."

N_COLOR="$(ls -1 "$OAK_DIR/color/"*.jpg 2>/dev/null | wc -l | tr -d ' ')"
N_DEPTH="$(ls -1 "$OAK_DIR/depth/"*.png 2>/dev/null | wc -l | tr -d ' ')"
info "$N_COLOR colour frames, $N_DEPTH depth frames"
[[ "$N_COLOR" -eq 0 ]] && die "No .jpg files in $OAK_DIR/color/ — capture may have failed."
[[ "$N_COLOR" -ne "$N_DEPTH" ]] && die "Colour/depth frame count mismatch ($N_COLOR vs $N_DEPTH)."

# ── Helper: run a Python step ─────────────────────────────────────────────
STEP_NUM=0
run_step() {
  local label="$1"; shift
  STEP_NUM=$((STEP_NUM + 1))
  banner "Step $STEP_NUM: $label"

  if $VERBOSE; then
    "$@"
  else
    local LOG
    LOG="$(mktemp)"
    if ! "$@" > "$LOG" 2>&1; then
      cat "$LOG" >&2
      rm -f "$LOG"
      die "Step $STEP_NUM ($label) FAILED. See output above."
    fi
    tail -n 8 "$LOG"
    rm -f "$LOG"
  fi
  info "$label — done"
}

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Open3D reconstruction
# ═══════════════════════════════════════════════════════════════════════════
RECON_ARGS=(
  python3 pipelines/3d/open3d_reconstruct.py
  --in "$OAK_DIR"
  --out-dir "$RUN_DIR/processed"
  --no-viz
)
[[ -n "$EVERY_N" ]]     && RECON_ARGS+=(--every-n "$EVERY_N")
[[ -n "$MAX_FRAMES" ]]  && RECON_ARGS+=(--max-frames "$MAX_FRAMES")
[[ -n "$DEPTH_TRUNC" ]] && RECON_ARGS+=(--depth-trunc "$DEPTH_TRUNC")
[[ -n "$RECON_VOXEL" ]] && RECON_ARGS+=(--voxel-size "$RECON_VOXEL")
[[ -n "$ODO_METHOD" ]]  && RECON_ARGS+=(--odometry-method "$ODO_METHOD")
$SAVE_GLB && RECON_ARGS+=(--save-glb)

run_step "Open3D reconstruction" "${RECON_ARGS[@]}"

[[ -f "$RUN_DIR/processed/trajectory.csv" ]] \
  || die "Reconstruction finished but trajectory.csv was not created."

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Clean geometry
# ═══════════════════════════════════════════════════════════════════════════
if ! $SKIP_CLEAN; then
  PCD_RAW="$RUN_DIR/processed/open3d_pcd_raw.ply"
  MESH_RAW="$RUN_DIR/processed/open3d_mesh_raw.ply"

  # Fallback: sample PCD from mesh if pcd_raw doesn't exist
  if [[ ! -f "$PCD_RAW" && -f "$MESH_RAW" ]]; then
    banner "Step 2a: Sampling point cloud from mesh (pcd_raw not found)"
    python3 -c "
import open3d as o3d, sys
m = o3d.io.read_triangle_mesh('$MESH_RAW')
if m.is_empty(): sys.exit(1)
n = min(200000, len(m.vertices))
pcd = m.sample_points_uniformly(number_of_points=n)
o3d.io.write_point_cloud('$PCD_RAW', pcd)
print(f'Sampled {len(pcd.points)} points from mesh -> $PCD_RAW')
" || warn "Could not sample pcd from mesh; cleaning will be skipped."
  fi

  if [[ -f "$PCD_RAW" ]]; then
    CLEAN_ARGS=(
      python3 pipelines/3d/clean_geometry.py
      --run-dir  "$RUN_DIR"
      --in-pcd   "$PCD_RAW"
      --out-pcd  "$RUN_DIR/processed/open3d_pcd_clean.ply"
      --out-mesh "$RUN_DIR/processed/open3d_mesh_clean.ply"
      --downsample       "$CLEAN_VOXEL"
      --sor-nb-neighbors "$CLEAN_SOR_NB"
      --sor-std-ratio    "$CLEAN_SOR_STD"
      --remove-plane
    )
    # Supply raw mesh for direct mesh cleaning path
    [[ -f "$MESH_RAW" ]] && CLEAN_ARGS+=(--in-mesh "$MESH_RAW")
    # Trajectory crop if trajectory exists
    if [[ -f "$RUN_DIR/processed/trajectory.csv" ]]; then
      CLEAN_ARGS+=(--trajectory "$RUN_DIR/processed/trajectory.csv" --crop-from-trajectory)
    else
      warn "No trajectory for crop; running clean_geometry without trajectory crop."
    fi
    $SAVE_GLB && CLEAN_ARGS+=(--save-glb)

    # If cleaning fails, warn but continue with raw outputs
    if ! run_step "Clean geometry" "${CLEAN_ARGS[@]}" 2>/dev/null; then
      warn "Geometry cleaning failed; continuing with raw outputs."
    fi
  else
    warn "No raw point cloud or mesh found; skipping cleaning step."
  fi
else
  info "Cleaning skipped (--skip-clean)"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Steps 3–4: Mag fusion + Voxel volume (skipped in camera-only mode)
# ═══════════════════════════════════════════════════════════════════════════
if $CAMERA_ONLY; then
  info "Steps 3–4 skipped (camera-only mode — no mag fusion / voxelisation)"
else
  # ── Step 3: Fuse magnetometer with trajectory ──────────────────────────
  FUSE_ARGS=(
    python3 pipelines/3d/fuse_mag_with_trajectory.py
    --trajectory "$RUN_DIR/processed/trajectory.csv"
    --mag        "$MAG_CSV"
    --out        "$RUN_DIR/processed/mag_world.csv"
    --value-type zero_mag
  )

  if [[ -n "$DEFAULT_EXTRINSICS" ]]; then
    FUSE_ARGS+=(--default-extrinsics "$DEFAULT_EXTRINSICS")
  elif [[ -f "$EXT_JSON" ]]; then
    FUSE_ARGS+=(--extrinsics "$EXT_JSON")
  fi

  run_step "Fuse mag with trajectory" "${FUSE_ARGS[@]}"

  [[ -f "$RUN_DIR/processed/mag_world.csv" ]] \
    || die "Fusion finished but mag_world.csv was not created."

  # ── Step 4: Voxel volume ───────────────────────────────────────────────
  MAG_WORLD="$RUN_DIR/processed/mag_world.csv"
  if [[ ! -f "$MAG_WORLD" && -f "$RUN_DIR/processed/mag_world_m.csv" ]]; then
    MAG_WORLD="$RUN_DIR/processed/mag_world_m.csv"
    warn "Using auto-scaled mag_world_m.csv"
  fi

  run_step "Voxel volume" \
    python3 pipelines/3d/mag_world_to_voxel_volume.py \
      --in       "$MAG_WORLD" \
      --out      "$RUN_DIR/exports/volume.npz" \
      --voxel-size "$VOXEL_SIZE" \
      --max-dim    "$MAX_DIM"

  [[ -f "$RUN_DIR/exports/volume.npz" ]] \
    || die "Voxelisation finished but volume.npz was not created."
fi

# ═══════════════════════════════════════════════════════════════════════════
# Viewer (optional)
# ═══════════════════════════════════════════════════════════════════════════
if ! $NO_VIEWER; then
  VIEWER_STEP=$((STEP_NUM + 1))
  banner "Step $VIEWER_STEP: Launching viewer"
  echo "  Close the viewer window when done to see the final summary."
  python3 pipelines/3d/view_scan_toggle.py \
    --run "$RUN_DIR" \
    --title "FluxSpace — $(basename "$RUN_DIR")"
  info "Viewer closed"
else
  info "Viewer skipped (--no-viewer)"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
_rel() {
  python3 -c "
from pathlib import Path
try: print(Path('$1').relative_to('$REPO_ROOT'))
except ValueError: print('$1')
" 2>/dev/null || echo "$1"
}

# Determine best geometry available
GEOM_LABEL="(none)"
if [[ -f "$RUN_DIR/processed/open3d_mesh_clean.ply" ]]; then
  GEOM_LABEL="processed/open3d_mesh_clean.ply"
elif [[ -f "$RUN_DIR/processed/open3d_mesh_raw.ply" ]]; then
  GEOM_LABEL="processed/open3d_mesh_raw.ply (uncleaned)"
elif [[ -f "$RUN_DIR/processed/open3d_pcd_clean.ply" ]]; then
  GEOM_LABEL="processed/open3d_pcd_clean.ply (point cloud)"
fi

PIPELINE_MODE="full (camera + magnetometer)"
$CAMERA_ONLY && PIPELINE_MODE="camera-only (geometry only)"

printf "\n"
printf "${_G}════════════════════════════════════════${_R}\n"
printf "${_G}  Done — 3D pipeline complete           ${_R}\n"
printf "${_G}════════════════════════════════════════${_R}\n"
printf "\n"
printf "  RUN_DIR      : %s\n" "$(_rel "$RUN_DIR")"
printf "  Mode         : %s\n" "$PIPELINE_MODE"
printf "  Geometry     : %s\n" "$GEOM_LABEL"
[[ -f "$RUN_DIR/processed/open3d_pcd_raw.ply" ]] && \
printf "  Raw pcd      : %s\n" "$(_rel "$RUN_DIR/processed/open3d_pcd_raw.ply")"
[[ -f "$RUN_DIR/processed/open3d_pcd_clean.ply" ]] && \
printf "  Clean pcd    : %s\n" "$(_rel "$RUN_DIR/processed/open3d_pcd_clean.ply")"
[[ -f "$RUN_DIR/processed/reconstruction_report.json" ]] && \
printf "  Recon report : %s\n" "$(_rel "$RUN_DIR/processed/reconstruction_report.json")"
[[ -f "$RUN_DIR/processed/cleaning_report.json" ]] && \
printf "  Clean report : %s\n" "$(_rel "$RUN_DIR/processed/cleaning_report.json")"
printf "  Trajectory   : %s\n" "$(_rel "$RUN_DIR/processed/trajectory.csv")"

if ! $CAMERA_ONLY; then
  MAG_OUT="$RUN_DIR/processed/mag_world.csv"
  [[ -f "$RUN_DIR/processed/mag_world_m.csv" ]] && MAG_OUT="$RUN_DIR/processed/mag_world_m.csv (scaled)"
  printf "  Mag world    : %s\n" "$(_rel "$MAG_OUT")"
  printf "  Volume       : %s\n" "$(_rel "$RUN_DIR/exports/volume.npz")"
fi

printf "\n"
printf "  Re-open viewer:\n"
printf "    python3 pipelines/3d/view_scan_toggle.py --run \"%s\"\n" "$(_rel "$RUN_DIR")"
printf "\n"
