# Explanation of `tools/2d/setup_pi.sh`

This document explains the 2D **Raspberry Pi setup** script that installs system packages, enables I2C, and configures a Python virtual environment with dependencies for the 2D pipeline and MMC5983MA sensor.

---

## Overview

**`./tools/2d/setup_pi.sh`** is intended to be run **once** on a Raspberry Pi (e.g. over SSH) after cloning the repo. It:

1. Installs system packages (git, python3-venv, python3-pip, i2c-tools, python3-smbus).
2. Enables I2C non-interactively via `raspi-config`.
3. Creates or reuses a venv at `~/fluxenv`.
4. Installs Python packages: numpy, pandas, matplotlib, sparkfun-qwiic, sparkfun-qwiic-mmc5983ma.

After the first run, a **reboot** is often required for I2C to take effect.

---

## What it does

- **\[1/4\] System packages:** `sudo apt update` and `apt install -y` for git, python3-venv, python3-pip, i2c-tools, python3-smbus.
- **\[2/4\] I2C:** `sudo raspi-config nonint do_i2c 0` (enable I2C).
- **\[3/4\] Venv:** If `~/fluxenv` does not exist, `python3 -m venv ~/fluxenv`. Then `source` it and `pip install -U pip wheel setuptools`.
- **\[4/4\] Python deps:** `pip install -U numpy pandas matplotlib` and `pip install -U sparkfun-qwiic sparkfun-qwiic-mmc5983ma`.
- **Done:** Prints quick verification commands (`source ~/fluxenv/bin/activate`, `python -c "import qwiic_mmc5983ma; ..."`, `i2cdetect -y 1`) and notes that a reboot may be needed.

---

## Example usage

```bash
cd ~/fluxspace-core
chmod +x tools/2d/setup_pi.sh
./tools/2d/setup_pi.sh
sudo reboot
# After reboot, SSH back in:
source ~/fluxenv/bin/activate
i2cdetect -y 1
python -c "import qwiic_mmc5983ma; print('qwiic_mmc5983ma OK')"
```

---

## Relation to 2D pipeline

- Use before running `pipelines/2d/mag_to_csv.py` or `mag_to_csv_v2` on the Pi. The venv and sensor libraries are required for magnetometer logging.
- **3D on Pi:** For 3D capture (and optional PyVista), run `./tools/3d/setup_pi.sh` instead or in addition; it uses the same `~/fluxenv` and adds pyvista. See [PIPELINE_3D.md](../3d/PIPELINE_3D.md) and [raspberry_pi_setup.md](../raspberry_pi_setup.md).
