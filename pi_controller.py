import time
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

from gpiozero import AngularServo, DigitalOutputDevice


SERVO_PIN       = 17
PUMP_PIN        = 27
CAMERA_INDEX    = 0
CAMERA_WARMUP   = 2.0       # seconds
CAMERA_FOV      = 62.2      # degrees (horizontal FOV of Pi Camera v2)
SERVO_MIN       = -90       # physical limits of your servo
SERVO_MAX       =  90
SERVO_MOVE_WAIT = 1.5       # seconds to let servo reach position
WATER_DURATION  = 3.0       # seconds per plant
MODEL_A_PATH    = "dih_model_a_results/runs/train2/weights/best.pt"
MODEL_B_PATH    = "model_b.tflite"
CLASS_NAMES_PATH= "class_names.txt"



class DIHRobot:

    def __init__(self):
        self.servo = AngularServo(SERVO_PIN, min_angle=SERVO_MIN, max_angle=SERVO_MAX)
        self.pump  = DigitalOutputDevice(PUMP_PIN)
        self.servo.angle = 0

        print("Loading Model A (pot detection)...")
        self.model_a = YOLO(MODEL_A_PATH)

        print("Loading Model B (plant classification)...")
        self.interpreter = Interpreter(MODEL_B_PATH)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with open(CLASS_NAMES_PATH) as f:
            self.class_names = [l.strip() for l in f]

        print("Warming up camera...")
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        time.sleep(CAMERA_WARMUP)
        print("Ready.\n")


    def capture_image(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera read failed.")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def detect_pots(self, image):
        """Model A — returns list of dicts with bbox and center_x, sorted left→right."""
        results = self.model_a.predict(image, verbose=False)
        pots = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                pots.append({
                    "bbox":     (x1, y1, x2, y2),
                    "center_x": (x1 + x2) / 2,
                })
        pots.sort(key=lambda p: p["center_x"])
        return pots

    def identify_plant(self, crop):
        """Model B — returns (class_name, confidence)."""
        img = crop.convert("RGB").resize((224, 224))
        inp = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        self.interpreter.set_tensor(self.input_details[0]["index"], inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        out = np.exp(out - out.max())
        out /= out.sum()
        idx = int(np.argmax(out))
        return self.class_names[idx], float(out[idx])

    def needs_water(self, crop, plant_class):
        """Model C placeholder — always True for now."""
        return True


    def aim(self, center_x, image_width):
        """
        Map pixel X position to servo angle.

        The camera sees CAMERA_FOV degrees across `image_width` pixels.
        Center pixel  → 0°  (straight ahead)
        Left edge     → -FOV/2
        Right edge    → +FOV/2
        """
        normalized = (center_x / image_width) - 0.5   # -0.5 … +0.5
        angle = normalized * CAMERA_FOV                 # -31.1 … +31.1  (for 62.2° FOV)
        angle = max(SERVO_MIN, min(SERVO_MAX, angle))
        print(f"  Aiming servo to {angle:.1f}°")
        self.servo.angle = angle
        time.sleep(SERVO_MOVE_WAIT)

    def water(self, duration=WATER_DURATION):
        print(f"  Watering for {duration}s...")
        self.pump.on()
        time.sleep(duration)
        self.pump.off()


    def run_cycle(self):
        print("=== DIH cycle start ===")

        image = self.capture_image()
        image_width = image.size[0]

        pots = self.detect_pots(image)
        if not pots:
            print("No plants detected — going back to sleep.")
            return

        print(f"Detected {len(pots)} pot(s).")

        for i, pot in enumerate(pots):
            x1, y1, x2, y2 = pot["bbox"]
            print(f"\n[Plant {i+1}/{len(pots)}]")

            crop = image.crop((x1, y1, x2, y2))

            name, conf = self.identify_plant(crop)
            print(f"  Species: {name} ({conf*100:.1f}%)")

            if self.needs_water(crop, name):
                self.aim(pot["center_x"], image_width)
                self.water()
            else:
                print("  Doesn't need water — skipping.")

        print("\nResetting servo to center.")
        self.servo.angle = 0
        time.sleep(1.0)

        self.cap.release()
        print("=== Cycle complete — sleeping. ===\n")



if __name__ == "__main__":
    robot = DIHRobot()
    robot.run_cycle()