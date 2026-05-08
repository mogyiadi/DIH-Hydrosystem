from gpiozero import OutputDevice
from time import sleep

# Define the GPIO pin your MOSFET SIG is connected to
PUMP_PIN = 17

# Initialize the pump
# active_high=True means sending power (3.3V) turns the MOSFET on
# initial_value=False ensures the pump stays OFF when the script starts
pump = OutputDevice(PUMP_PIN, active_high=True, initial_value=False)

print("Starting water pump test...")

try:
    # 1. Turn the pump ON
    print("Pump is ON. Pumping water...")
    pump.on()

    # Let it run for 3 seconds
    sleep(3)

    # 2. Turn the pump OFF
    print("Pump is OFF.")
    pump.off()

    print("Test complete!")

except KeyboardInterrupt:
    # This catches the event if you press Ctrl+C to manually stop the script
    print("\nTest interrupted by user.")

finally:
    # SAFETY NET: This block always runs when the script ends or crashes.
    # It ensures you don't accidentally leave the pump running forever!
    print("Cleaning up: Ensuring pump is turned off.")
    pump.off()