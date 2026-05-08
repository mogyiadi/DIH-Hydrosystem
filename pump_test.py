import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
GPIO.output(17, GPIO.HIGH)
input('Pump should be ON now — press Enter to turn off...')
GPIO.output(17, GPIO.LOW)
GPIO.cleanup()