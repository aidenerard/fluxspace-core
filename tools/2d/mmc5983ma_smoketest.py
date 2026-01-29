from smbus2 import SMBus
import time

I2C_BUS = 1
MMC5983MA_ADDR = 0x30

def main():
    with SMBus(I2C_BUS) as bus:
        # Simple "ping": attempt a harmless read from register 0x00
        # If wiring is good, this should not throw an exception.
        _ = bus.read_byte_data(MMC5983MA_ADDR, 0x00)
        print("✅ MMC5983MA responding on I2C at address 0x30")

        # Optional: short pause so you can see it
        time.sleep(0.1)

if __name__ == "__main__":
    main()
