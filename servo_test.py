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


print(f"\n=== Setting Tilt Servo (Channel {CHANNEL}) ===")

try:
    # In pi_controller.py, the default level tilt is 6200 qms (1550 us)
    print("\n--- Moving to Level Position (1550 µs) ---")
    set_target_us(CHANNEL, 1550)
    time.sleep(2)

    # Almost upstraight. Depending on how the servo is physically mounted,
    # "up" will either be towards ~800 us or towards ~2200 us.
    print("\n--- Moving to Almost Upstraight Position (2200 µs) ---")
    print("(Note: If this points down instead of up, simply change 2200 to 800 in the script!)")
    set_target_us(CHANNEL, 200)

    print("\nPosition set. Holding... Press Ctrl+C to stop.")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nInterrupted! Leaving servo in almost vertical position and cleaning up...")
    time.sleep(1)
    port.close()
    print("Port closed.")

