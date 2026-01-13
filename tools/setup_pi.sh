#!/usr/bin/env bash
set -euo pipefail

echo "=== Fluxspace Pi setup (system + python) ==="

# ---- 1) System packages ----
echo "[1/4] Installing system packages..."
sudo apt update
sudo apt install -y \
  git \
  python3-venv python3-pip \
  i2c-tools \
  python3-smbus

# ---- 2) Enable I2C (non-interactive) ----
# 0 = enable, 1 = disable
echo "[2/4] Enabling I2C..."
sudo raspi-config nonint do_i2c 0

# ---- 3) Create / reuse venv ----
VENV_PATH="${HOME}/fluxenv"
echo "[3/4] Creating/using venv at: ${VENV_PATH}"
if [ ! -d "${VENV_PATH}" ]; then
  python3 -m venv "${VENV_PATH}"
fi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"
python -m pip install -U pip wheel setuptools

# ---- 4) Python deps (pipeline + sensor) ----
echo "[4/4] Installing Python packages..."
pip install -U numpy pandas matplotlib
pip install -U sparkfun-qwiic sparkfun-qwiic-mmc5983ma

echo ""
echo "=== Done. Quick checks: ==="
echo "  source ~/fluxenv/bin/activate"
echo "  python -c \"import qwiic_mmc5983ma; print('qwiic_mmc5983ma OK')\""
echo "  i2cdetect -y 1"
echo ""
echo "NOTE: I2C enable may require a reboot the first time."
echo "Reboot now if needed: sudo reboot"
