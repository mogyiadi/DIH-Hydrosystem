import time
import serial

# --- Configuration ---
CHANNEL = 2  # Changed to Tilt servo (was 1 for pan)
PORT = '/dev/ttyACM0'
BAUD_RATE = 9600

print("Connecting to Maestro controller...")
try:
    port = serial.Serial(PORT, BAUD_RATE, timeout=1)
    print("Connected to controller!")
except Exception as e:
    print(f"Could not connect to serial port. Error: {e}")
    exit()


def set_target_us(channel, target_us):
    """
    Sends target to Maestro using standard microseconds (500 to 2500).
    Automatically converts to Maestro's required quarter-microseconds.
    """
    # Convert microseconds to quarter-microseconds
    target_qms = int(target_us * 4)

    lsb = target_qms & 0x7F
    msb = (target_qms >> 7) & 0x7F
    cmd = bytes([0x84, channel, lsb, msb])
    port.write(cmd)

    print(f"Sent {target_us} µs (Maestro target: {target_qms}) to Channel {channel}")


print(f"\n=== Setting Servos to Straight Position ===")

try:
    print("\n--- Moving Channel 0 to Straight Position (1500 µs) ---")
    set_target_us(0, 100)
    time.sleep(1)

    print("\n--- Moving Channel 2 to Straight Position (1500 µs) ---")
    set_target_us(2, 100)

    print("\nPositions set. Holding... Press Ctrl+C to stop.")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nInterrupted! Leaving servos in straight position and cleaning up...")
    time.sleep(1)
    port.close()
    print("Port closed.")

