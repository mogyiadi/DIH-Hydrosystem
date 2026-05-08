"""
Pump control script.
Controls a water pump via MOSFET on GPIO pin 17.

Keys:
  SPACE  — pump on, press any key to stop
  p      — pulse: run pump for set duration then stop automatically
  +      — increase pulse duration by 0.5s
  -      — decrease pulse duration by 0.5s
  q      — quit
"""

import sys
import termios
import tty
import time
import RPi.GPIO as GPIO

PUMP_PIN       = 17
PULSE_DURATION = 2.0  # seconds, adjustable with +/-


def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PUMP_PIN, GPIO.OUT, initial=GPIO.LOW)
    print(f"Pump ready on GPIO {PUMP_PIN}.")


def pump_on():
    GPIO.output(PUMP_PIN, GPIO.HIGH)


def pump_off():
    GPIO.output(PUMP_PIN, GPIO.LOW)


def cleanup():
    pump_off()
    GPIO.cleanup()
    print("GPIO cleaned up.")


def getch():
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    global PULSE_DURATION
    setup()

    print(__doc__)
    print(f"Pulse duration: {PULSE_DURATION:.1f}s\n")

    try:
        while True:
            ch = getch()

            if ch == 'q':
                print("\nQuitting.")
                break

            elif ch == ' ':
                print("  Pump ON — press any key to stop...", end="", flush=True)
                pump_on()
                getch()  # block until next keypress
                pump_off()
                print("  OFF")

            elif ch == 'p':
                print(f"  Pulsing for {PULSE_DURATION:.1f}s...")
                pump_on()
                time.sleep(PULSE_DURATION)
                pump_off()
                print("  Done.")

            elif ch == '+':
                PULSE_DURATION = round(PULSE_DURATION + 0.5, 1)
                print(f"  Pulse duration: {PULSE_DURATION:.1f}s")

            elif ch == '-':
                PULSE_DURATION = max(0.5, round(PULSE_DURATION - 0.5, 1))
                print(f"  Pulse duration: {PULSE_DURATION:.1f}s")

    finally:
        cleanup()


if __name__ == "__main__":
    main()