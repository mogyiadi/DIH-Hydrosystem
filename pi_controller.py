import time
import serial
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from picamera2 import Picamera2

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


CAMERA_FOV      = 62.2
SERVO_MOVE_WAIT = 1.5
MODEL_A_PATH    = "dih_model_a_results/runs/train2/weights/best.pt"
MODEL_B_PATH    = "model_b.tflite"
CLASS_NAMES_PATH= "class_names.txt"



class DIHRobot:

    def __init__(self):
        print("Connecting to Maestro controller...")
        try:
            self.port = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
            print("Connected to controller!")

            # Home arm to your default position
            self.set_target(0, 7000)
            self.set_target(1, 6000)
            self.set_target(2, 6200)
            self.current_pan = 6000
            time.sleep(1)
        except Exception as e:
            print(f"Could not connect to serial port. Error: {e}")
            self.port = None

        print("Loading Model A (pot detection)...")
        self.model_a = YOLO(MODEL_A_PATH)

        print("Loading Model B (plant classification)...")
        self.interpreter = Interpreter(MODEL_B_PATH)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with open(CLASS_NAMES_PATH) as f:
            self.class_names = [l.strip() for l in f]

        print("Warming up camera natively with Picamera2...")
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2.0)  # Let the camera auto-expose
        print("Ready.\n")

    def set_target(self, channel, target):
        """Send target to Maestro (target in quarter-microseconds)."""
        if not self.port:
            return

        lsb = target & 0x7F
        msb = (target >> 7) & 0x7F
        cmd = bytes([0x84, channel, lsb, msb])
        self.port.write(cmd)

    def capture_image(self):
        """Capture from Picamera2 and return a PIL Image in RGB format."""
        frame = self.picam2.capture_array()

        # Safety Check: If the frame somehow still has 4 channels, slice off the 4th one
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]

        return Image.fromarray(frame)

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
                    "center_y": (y1 + y2) / 2,
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

    def aim(self, center_x, center_y, image_width, image_height):
        """
        Map pixel positions to Maestro targets for Servos 1 (Pan) and 2 (Tilt).
        """
        # Horizontal / Pan (Servo 1)
        normalized_x = (center_x / image_width) - 0.5
        angle_x = normalized_x * CAMERA_FOV  # Uses horizontal FOV (62.2°)

        # Vertical / Tilt (Servo 2)
        CAMERA_FOV_V = 48.8  # Vertical FOV for Pi Camera v2
        normalized_y = (center_y / image_height) - 0.5
        angle_y = normalized_y * CAMERA_FOV_V

        print(f"  Target requires a horizontal shift of {angle_x:.1f}° and vertical shift of {angle_y:.1f}°.")

        # Map degrees to Maestro target (1 degree ~22.2 qms)
        # Default centers: Servo 1 = 6000, Servo 2 = 6200
        target_1_qms = self.current_pan + int(angle_x * 22.22)
        target_2_qms = 6200 + int(angle_y * 22.22)

        # Apply hard safety limits to prevent the servos from over-rotating
        target_1_qms = max(4000, min(8000, target_1_qms))
        target_2_qms = max(4000, min(8000, target_2_qms))

        print(f"  Sending Pan  (Servo 1) to {target_1_qms}")
        print(f"  Sending Tilt (Servo 2) to {target_2_qms}")

        self.set_target(1, target_1_qms)
        self.set_target(2, target_2_qms)
        time.sleep(SERVO_MOVE_WAIT)

    def run_cycle(self):
        print("=== DIH cycle start ===")

        pan_steps = [4000, 5000, 6000, 7000, 8000]

        for pan_pos in pan_steps:
            print(f"\n--- Scanning at pan position {pan_pos} ---")
            self.set_target(1, pan_pos)
            self.set_target(2, 6200)  # Ensure tilt is level for the scan
            self.current_pan = pan_pos
            time.sleep(SERVO_MOVE_WAIT)

            image = self.capture_image()
            image_width = image.size[0]
            image_height = image.size[1]

            pots = self.detect_pots(image)
            if not pots:
                print("No plants detected — going back to sleep.")
            else:
                print(f"Detected {len(pots)} pot(s).")
                for i, pot in enumerate(pots):
                    x1, y1, x2, y2 = pot["bbox"]
                    print(f"\n[Plant {i + 1}/{len(pots)}]")

                    crop = image.crop((x1, y1, x2, y2))
                    name, conf = self.identify_plant(crop)
                    print(f"  Species: {name} ({conf * 100:.1f}%)")

                    if self.needs_water(crop, name):
                        self.aim(pot["center_x"], pot["center_y"], image_width, image_height)
                        # No water logic for now since pump is disconnected
                        print("  *Pretending to water*")
                        time.sleep(2.0)

                        print("  Returning to scan position for next plant...")
                        self.set_target(1, self.current_pan)
                        self.set_target(2, 6200)
                        time.sleep(SERVO_MOVE_WAIT)
                    else:
                        print("  Doesn't need water — skipping.")

        print("\nResetting arm to homed position.")
        self.set_target(0, 7000)
        self.set_target(1, 6000)
        self.set_target(2, 6200)
        time.sleep(1.0)

        self.cleanup()
        print("=== Cycle complete — sleeping. ===\n")

    def cleanup(self):
        self.picam2.stop()
        if self.port:
            self.port.close()


if __name__ == "__main__":
    robot = DIHRobot()
    try:
        robot.run_cycle()
    except KeyboardInterrupt:
        print("Interrupted! Cleaning up...")
        robot.cleanup()