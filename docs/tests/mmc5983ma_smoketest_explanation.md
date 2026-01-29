# Explanation of `mmc5983ma_smoketest.py`

This document explains the MMC5983MA sensor smoke test script.

---

## Overview

This script performs a simple connectivity test to verify that the MMC5983MA magnetometer sensor is properly connected and responding on the I2C bus. It's a quick diagnostic tool to check if your hardware setup is working before running the main data collection pipeline.

**What it does:**
- Connects to the I2C bus
- Attempts to read from the MMC5983MA sensor at address 0x30
- Prints a success message if the sensor responds
- Exits with an error if the sensor is not found

**Typical usage:**
```bash
python3 tools/2d/mmc5983ma_smoketest.py
```

**Expected output:**
- `✅ MMC5983MA responding on I2C at address 0x30` (if successful)
- An exception/error if the sensor is not connected or not responding

---

## Code Explanation

### Imports and Constants (Lines 1-5)

```python
from smbus2 import SMBus
import time

I2C_BUS = 1
MMC5983MA_ADDR = 0x30
```

**What it does:**
- `smbus2.SMBus`: Python library for I2C communication on Linux (used on Raspberry Pi)
- `time`: Used for a short pause after successful detection
- `I2C_BUS = 1`: Specifies I2C bus 1 (standard on Raspberry Pi)
- `MMC5983MA_ADDR = 0x30`: The I2C address of the MMC5983MA sensor (0x30 in hexadecimal)

### Main Function (Lines 7-15)

```python
def main():
    with SMBus(I2C_BUS) as bus:
        # Simple "ping": attempt a harmless read from register 0x00
        # If wiring is good, this should not throw an exception.
        _ = bus.read_byte_data(MMC5983MA_ADDR, 0x00)
        print("✅ MMC5983MA responding on I2C at address 0x30")
        time.sleep(0.1)
```

**What it does:**
1. Opens I2C bus 1 using a context manager (automatically closes when done)
2. Attempts to read one byte from register 0x00 on the sensor
3. If successful, prints a confirmation message
4. If the sensor is not found or not responding, an exception is raised and the script exits
5. Adds a 0.1 second pause so the message is visible

**Why register 0x00?**
- Register 0x00 is typically a safe register to read (often contains device ID or status)
- This is a "harmless" read that won't affect sensor operation
- If the sensor is connected, this read will succeed; if not, it will fail

---

## When to Use This Script

Use this script:
- After first-time hardware setup to verify I2C connection
- When troubleshooting sensor connectivity issues
- Before running `mag_to_csv.py` to ensure the sensor is ready
- After making changes to wiring or hardware configuration

**Common scenarios:**
- **Success**: Sensor is connected and ready for data collection
- **Failure**: Check wiring, I2C enablement (`raspi-config`), and sensor power

---

## Integration with Setup Workflow

This script is referenced in the Raspberry Pi setup runbook (Part D2) as an optional smoke test after confirming the sensor appears on I2C with `i2cdetect -y 1`.

**Typical workflow:**
1. Run `i2cdetect -y 1` to see if address 0x30 appears
2. Run `python3 tools/2d/mmc5983ma_smoketest.py` to verify the sensor responds
3. Proceed with `mag_to_csv.py` for data collection

---

## Dependencies

- `smbus2`: Python library for I2C communication
  - Install with: `pip install smbus2`
- I2C must be enabled on Raspberry Pi (via `raspi-config`)
- Sensor must be physically connected to I2C bus 1

---

## Error Handling

If the sensor is not found, the script will raise an exception (typically `OSError` or `IOError`) and exit. This indicates:
- Sensor is not connected
- Wrong I2C address
- I2C bus not enabled
- Wiring issues
- Sensor not powered

