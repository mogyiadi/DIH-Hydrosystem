import time
import serial

# --- Configuration ---
CHANNEL = 1
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


print(f"\n=== Starting Limit Test on Channel {CHANNEL} ===")

try:
    while True:
        print("\n--- Moving to Minimum limit (500 µs) ---")
        # Watch the starting position
        set_target_us(CHANNEL, 500)
        time.sleep(4)

        print("\n--- Moving to Maximum limit (2500 µs) ---")
        # Watch how far it travels from the start
        set_target_us(CHANNEL, 2500)
        time.sleep(4)

        print("\n--- Returning to Center (1500 µs) ---")
        set_target_us(CHANNEL, 1500)
        time.sleep(4)

        print("\nRepeating test. Press Ctrl+C to stop.")

except KeyboardInterrupt:
    print("\nInterrupted! Returning to center and cleaning up...")
    set_target_us(CHANNEL, 1500)
    time.sleep(1)
    port.close()
    print("Port closed.")