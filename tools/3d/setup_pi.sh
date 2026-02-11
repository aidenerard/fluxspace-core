#!/usr/bin/env bash
set -euo pipefail

# ---- Parse flags ----
INSTALL_OAKD=false
for arg in "$@"; do
  case "$arg" in
    --with-oakd) INSTALL_OAKD=true ;;
    --help|-h)
      echo "Usage: ./tools/3d/setup_pi.sh [--with-oakd]"
      echo ""
      echo "Sets up the Fluxspace 3D pipeline on a Raspberry Pi."
      echo ""
      echo "Options:"
      echo "  --with-oakd   Also install OAK-D Lite / Open3D deps"
      echo "                (depthai, opencv-python, open3d)"
      exit 0
      ;;
  esac
done

echo "=== Fluxspace Pi setup — 3D pipeline (system + Python) ==="

# ---- 1) System packages ----
echo "[1/5] Installing system packages..."
SYSTEM_PKGS=(
  git
  python3-venv python3-pip
  i2c-tools
  python3-smbus
  python3-vtk9
  build-essential cmake pkg-config python3-dev
)
# libusb is required by depthai (OAK-D SDK) on Linux
if $INSTALL_OAKD; then
  SYSTEM_PKGS+=( libusb-1.0-0-dev )
fi
sudo apt update
sudo apt install -y "${SYSTEM_PKGS[@]}"

# ---- 2) Enable I2C (non-interactive) ----
echo "[2/5] Enabling I2C..."
sudo raspi-config nonint do_i2c 0

# ---- 3) Create / reuse venv ----
VENV_PATH="${HOME}/fluxenv"
echo "[3/5] Creating/using venv at: ${VENV_PATH}"
if [ ! -d "${VENV_PATH}" ]; then
  python3 -m venv "${VENV_PATH}"
fi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"
python -m pip install -U pip wheel setuptools

# ---- 4) Python deps (2D + 3D pipeline + sensor) ----
echo "[4/5] Installing Python packages (2D + 3D + magnetometer)..."
pip install -U numpy pandas matplotlib scipy scikit-learn
pip install -U sparkfun-qwiic sparkfun-qwiic-mmc5983ma
# 3D: voxel heatmap visualization (optional on Pi; often run on Mac)
pip install -U pyvista

# ---- 5) OAK-D Lite + Open3D (optional) ----
if $INSTALL_OAKD; then
  echo "[5/5] Installing OAK-D + Open3D packages..."
  # capture_oak_rgbd.py: DepthAI SDK + OpenCV for OAK-D RGB-D capture
  pip install -U depthai opencv-python
  # open3d_reconstruct.py: Open3D for TSDF reconstruction from captured frames
  pip install -U open3d
else
  echo "[5/5] Skipping OAK-D + Open3D (pass --with-oakd to install)."
fi

echo ""
echo "=== Done. Quick checks: ==="
echo "  source ~/fluxenv/bin/activate"
echo "  python -c \"import qwiic_mmc5983ma; print('qwiic_mmc5983ma OK')\""
echo "  python -c \"import pyvista; print('pyvista OK')\""
echo "  python -c \"import vtk; print('vtk OK')\""
echo "  i2cdetect -y 1"
if $INSTALL_OAKD; then
  echo ""
  echo "  OAK-D checks:"
  echo "  python -c \"import depthai; print('depthai OK')\""
  echo "  python -c \"import cv2; print('opencv OK')\""
  echo "  python -c \"import open3d; print('open3d OK')\""
fi
echo ""
echo "NOTE: I2C enable may require a reboot the first time."
echo "Reboot now if needed: sudo reboot"
