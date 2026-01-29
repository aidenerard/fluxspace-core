# Explanation of `tools/2d/mmc5983ma_smoketest.py`

This document explains the **MMC5983MA I2C smoketest** script used to verify that the magnetometer is wired correctly and visible on the I2C bus.

---

## Overview

**`python3 tools/2d/mmc5983ma_smoketest.py`** performs a minimal read from the MMC5983MA at I2C address `0x30` (register `0x00`). If the read succeeds, it prints a success message; otherwise it raises, which helps catch wiring or I2C configuration issues before running the full pipeline.

**Dependencies:** `smbus2` (often via `python3-smbus` on the Pi). The 2D Pi setup (`./tools/2d/setup_pi.sh`) does not install `smbus2` by default; you may need `pip install smbus2` or use a system Python that provides it.

---

## What it does

1. Opens I2C bus 1 (`/dev/i2c-1` on typical Pi).
2. Calls `bus.read_byte_data(MMC5983MA_ADDR, 0x00)` (MMC5983MA at `0x30`). This is a low-level register read; no decoding of magnetic values.
3. On success: prints `✅ MMC5983MA responding on I2C at address 0x30`, sleeps 0.1 s, and exits 0.
4. On I2C error (e.g. NACK, missing device): the `SMBus` context manager raises; the script exits non‑zero.

---

## Example usage

```bash
cd ~/fluxspace-core
source ~/fluxenv/bin/activate   # if using venv
python3 tools/2d/mmc5983ma_smoketest.py
```

Use this after Pi setup and wiring to confirm the sensor is detected. See [raspberry_pi_setup.md](../raspberry_pi_setup.md) and [setup_pi_explanation.md](setup_pi_explanation.md). For full pipeline usage, see [mag_to_csv_explanation.md](mag_to_csv_explanation.md) and [PIPELINE_2D.md](PIPELINE_2D.md).
