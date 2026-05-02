"""
Servo calibration helper.
Controls servo 0 and servo 2 interactively so you can measure
the physical angle at known QMS values.

Keys:
  W / S  — servo 0 up / down  (large step)
  w / s  — servo 0 up / down  (small step)
  I / K  — servo 2 up / down  (large step)
  i / k  — servo 2 up / down  (small step)
  0      — send both servos to their VERTICAL constants
  space  — print current values (copy these into your notes)
  q      — quit
"""

import sys
import termios
import tty
import serial
import time

SERVO_0_VERTICAL_POS = 8000
SERVO_2_VERTICAL_POS = 4000

LARGE_STEP = 500   # ~a few degrees, adjust if needed
SMALL_STEP = 100   # fine tune

def set_target(port, channel, target):
    target = max(0, min(16000, target))
    lsb = target & 0x7F
    msb = (target >> 7) & 0x7F
    port.write(bytes([0x84, channel, lsb, msb]))
    return target

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def main():
    print("Connecting...")
    port = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(0.5)

    s0 = SERVO_0_VERTICAL_POS
    s2 = SERVO_2_VERTICAL_POS

    set_target(port, 0, s0)
    set_target(port, 2, s2)
    time.sleep(1.0)

    print(f"Starting at  Servo0={s0}  Servo2={s2}")
    print(__doc__)

    while True:
        ch = getch()

        if ch == 'q':
            print("\nDone.")
            break
        elif ch == 'W':
            s0 = set_target(port, 0, s0 + LARGE_STEP)
        elif ch == 'S':
            s0 = set_target(port, 0, s0 - LARGE_STEP)
        elif ch == 'w':
            s0 = set_target(port, 0, s0 + SMALL_STEP)
        elif ch == 's':
            s0 = set_target(port, 0, s0 - SMALL_STEP)
        elif ch == 'I':
            s2 = set_target(port, 2, s2 + LARGE_STEP)
        elif ch == 'K':
            s2 = set_target(port, 2, s2 - LARGE_STEP)
        elif ch == 'i':
            s2 = set_target(port, 2, s2 + SMALL_STEP)
        elif ch == 'k':
            s2 = set_target(port, 2, s2 - SMALL_STEP)
        elif ch == '0':
            s0 = set_target(port, 0, SERVO_0_VERTICAL_POS)
            s2 = set_target(port, 2, SERVO_2_VERTICAL_POS)
        elif ch == ' ':
            pass  # just print below

        print(f"  Servo0={s0}  Servo2={s2}")

    port.close()

if __name__ == "__main__":
    main()