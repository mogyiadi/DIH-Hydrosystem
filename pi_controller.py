import time
import serial
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from picamera2 import Picamera2
import math

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter


CAMERA_FOV      = 48.8
CAMERA_FOV_V    = 62.2
SERVO_MOVE_WAIT = 1.5
# MODEL_A_PATH    = "dih_model_a_results/runs/train2/weights/best.pt"
MODEL_A_PATH    = "yolo26n.pt"
MODEL_B_PATH    = "model_b.tflite"
CLASS_NAMES_PATH= "class_names.txt"

# How close two pan positions must be (in quarter-microseconds) to be considered
# the same physical plant.  The scan step is 100 qms; a plant is typically visible
# across ~5-10 steps, so 600 qms gives a safe margin without merging neighbours.
DEDUP_THRESHOLD = 600

# Vertical position for arm base (Servo 0) and top tilt (Servo 2)
SERVO_0_VERTICAL_POS = 8000
SERVO_2_VERTICAL_POS = 4000

# Bow position for watering (Hose points forward/down)
SERVO_0_BOW_POS = 5000
SERVO_2_BOW_POS = 4000

# Tilt levels for scanning to capture both near and far plants
TILT_STEPS = [4000, 5000, 6000]

CAMERA_HEIGHT_CM = 22.7

# Calibrated conversions
S0_REF_QMS, S0_REF_DEG = 8000, 2.0
S0_QMS_PER_DEG = 40.7  # increasing QMS = more vertical

S2_REF_QMS, S2_REF_DEG = 4000, 17.0
S2_QMS_PER_DEG = 42.5  # increasing QMS = more horizontal

class DIHRobot:

    def __init__(self):
        print("Connecting to Maestro controller...")
        try:
            self.port = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
            print("Connected to controller!")

            # Home arm to your default position (vertical)
            self.set_target(0, SERVO_0_VERTICAL_POS)
            self.set_target(1, 6000)
            self.set_target(2, SERVO_2_VERTICAL_POS)
            self.current_pan = 6000
            self.current_tilt = SERVO_2_VERTICAL_POS
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
        
        # Improve color accuracy
        self.picam2.set_controls({"AwbEnable": False, "AwbMode": 1})  # Use daylight white balance

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

        # Camera is rotated 90 degrees to the right (clockwise)
        # To correct, rotate 90 degrees counter-clockwise (expand=True to swap dimensions)
        return Image.fromarray(frame).rotate(90, expand=True)

    def detect_pots(self, image):
        """Model A — returns list of dicts with bbox and center_x, sorted left→right."""
        # We only want to identify class 58 (potted plant)
        results = self.model_a.predict(image, conf=0.25, classes=[58], verbose=False)
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
        Iteratively center the plant in the frame.
        """
        cx, cy = center_x, center_y
        final_img = None
        final_pot = None

        for _ in range(5):  # Max 5 iterations to center
            # Horizontal / Pan (Servo 1)
            normalized_x = (cx / image_width) - 0.5
            angle_x = normalized_x * CAMERA_FOV

            # Vertical / Tilt (Servo 2)
            normalized_y = (cy / image_height) - 0.5
            angle_y = normalized_y * CAMERA_FOV_V

            if abs(normalized_x) < 0.05 and abs(normalized_y) < 0.05:
                print("  Target centered.")
                break

            print(f"  Target requires a horizontal shift of {angle_x:.1f}° and vertical shift of {angle_y:.1f}°.")

            # update corrections (flipped sign for tilt to look down correctly)
            target_1_qms = self.current_pan  - int(angle_x * 25.0)
            target_2_qms = self.current_tilt + int(angle_y * 25.0)

            target_1_qms = max(0, min(16000, target_1_qms))
            target_2_qms = max(0, min(16000, target_2_qms))

            self.set_target(1, target_1_qms)
            self.set_target(2, target_2_qms)
            self.current_pan = target_1_qms
            self.current_tilt = target_2_qms

            time.sleep(1.0)

            # Recapture and get new center
            img = self.capture_image()
            pots = self.detect_pots(img)

            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.putText(cv_img, "Aiming...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            for pot in pots:
                x1, y1, x2, y2 = pot["bbox"]
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 128), 2)
                cv2.putText(cv_img, "Pot", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 128), 2)

            cv2.imshow("Live Feed", cv_img)
            cv2.waitKey(1)

            if not pots:
                break

            # Find the pot closest to the center of the image
            closest = min(pots, key=lambda p: abs(p["center_x"]/image_width - 0.5) + abs(p["center_y"]/image_height - 0.5))
            cx, cy = closest["center_x"], closest["center_y"]
            final_img = img
            final_pot = closest

        return final_img, final_pot

    def tilt_qms_to_deg(self, qms):
        """Convert servo 2 QMS to physical angle in degrees (0=vertical, 90=horizontal)."""
        return S2_REF_DEG + (qms - S2_REF_QMS) / S2_QMS_PER_DEG

    def deg_to_s0_qms(self, deg):
        """Convert desired angle to servo 0 QMS. Clamp to physical limits."""
        qms = S0_REF_QMS - int((deg - S0_REF_DEG) * S0_QMS_PER_DEG)
        return max(4400, min(8000, qms))

    def deg_to_s2_qms(self, deg):
        """Convert desired angle to servo 2 QMS. Clamp to physical limits."""
        qms = S2_REF_QMS + int((deg - S2_REF_DEG) * S2_QMS_PER_DEG)
        return max(4000, min(7100, qms))

    def compute_bow(self):
        """
        The camera is centered on the plant at self.current_tilt.
        The hose points 90° away from the camera, so we add 90° to the
        current tilt angle to get the bow position.

        We also use trig to estimate distance and nudge servo 0 forward
        so the hose tip moves closer to the plant if needed.
        """
        current_angle_deg = self.tilt_qms_to_deg(self.current_tilt)
        bow_angle_deg = current_angle_deg + 90.0

        # Distance estimate — used to offset servo 0
        tilt_rad = math.radians(current_angle_deg)
        distance_cm = CAMERA_HEIGHT_CM / math.tan(tilt_rad) if tilt_rad > 0 else 999

        print(f"  Camera tilt: {current_angle_deg:.1f}°  →  bow angle: {bow_angle_deg:.1f}°")
        print(f"  Estimated plant distance: {distance_cm:.1f} cm")

        s2_bow = self.deg_to_s2_qms(bow_angle_deg)

        # Distance → servo 0: at ≤35 cm stay fully vertical (don't crush plant),
        # lean progressively forward for farther plants.
        s0_bow = int(np.interp(distance_cm,
                               [30, 90],
                               [8000, 5000]))
        s0_bow = max(4400, min(8000, s0_bow))

        return s0_bow, s2_bow

    def run_cycle(self):
        print("=== DIH cycle start ===")

        # Sweep forward and backward
        forward_steps = list(range(4000, 7501, 100))

        recent_plants = []  # list of (centred_pan, timestamp)

        while True:
            for i, tilt_pos in enumerate(TILT_STEPS):
                print(f"\n=== Scanning at tilt position {tilt_pos} ===")
                
                # Alternate direction per tilt level (zigzag)
                current_pan_steps = forward_steps if i % 2 == 0 else list(reversed(forward_steps))
                
                for pan_pos in current_pan_steps:
                    print(f"\n--- Scanning at pan {pan_pos}, tilt {tilt_pos} ---")
                    self.set_target(1, pan_pos)
                    self.set_target(2, tilt_pos)
                    self.current_pan = pan_pos
                    self.current_tilt = tilt_pos

                    time.sleep(0.1)

                    image = self.capture_image()
                    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                    image_width = image.size[0]
                    image_height = image.size[1]

                    pots = self.detect_pots(image)

                    # Draw all Model A bounding boxes early so we see them immediately
                    for pot in pots:
                        x1, y1, x2, y2 = pot["bbox"]
                        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 128), 2)
                        cv2.putText(cv_img, "Pot", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 128), 2)

                    cv2.imshow("Live Feed", cv_img)
                    cv2.waitKey(1)

                    if not pots:
                        print("No plants detected — moving on.")
                    else:
                        print(f"Detected {len(pots)} pot(s).")
                        for i, pot in enumerate(pots):
                            # Estimate the absolute pan that would centre this pot.
                            # Used only to decide whether we've already handled it.
                            normalized_x = (pot["center_x"] / image_width) - 0.5
                            angle_x = normalized_x * CAMERA_FOV
                            estimated_pan = self.current_pan - int(angle_x * 25.0)

                            if any(abs(estimated_pan - p[0]) < DEDUP_THRESHOLD for p in recent_plants):
                                print(f"  [Plant {i+1}] Already handled — skipping.")
                                continue

                            print(f"\n[Plant {i + 1}/{len(pots)}]")

                            # Aim and center this pot first
                            centered_img, centered_pot = self.aim(pot["center_x"], pot["center_y"], image_width, image_height)

                            if centered_pot is None:
                                print("  Lost plant during aiming.")
                                self.set_target(1, pan_pos)
                                self.set_target(2, tilt_pos)
                                self.current_pan = pan_pos
                                self.current_tilt = tilt_pos
                                continue

                            x1, y1, x2, y2 = centered_pot["bbox"]

                            # Expand crop slightly to give Model B better context
                            padding = 20
                            crop = centered_img.crop((x1 - padding, y1 - padding, x2 + padding, y2 + padding))

                            name, conf = self.identify_plant(crop)
                            print(f"  Species: {name} ({conf * 100:.1f}%)")

                            # We need to take a fresh feed frame and write the result on it
                            display_img = self.capture_image()
                            cv_display_img = cv2.cvtColor(np.array(display_img), cv2.COLOR_RGB2BGR)

                            if conf < 0.2:
                                print("  Confidence too low (<20%) — skipping.")
                                cv2.putText(cv_display_img, f"{name} {conf*100:.1f}% (LOW)", (x1, max(40, y1+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                cv2.imshow("Live Feed", cv_display_img)
                                cv2.waitKey(1)
                                # Returning to scan position
                                print("  Returning to scan position for next plant...")
                                self.set_target(1, pan_pos)
                                self.set_target(2, tilt_pos)
                                self.current_pan = pan_pos
                                self.current_tilt = tilt_pos
                                time.sleep(1.5)
                                continue

                            cv2.putText(cv_display_img, f"{name} {conf*100:.1f}%", (x1, max(40, y1+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.imshow("Live Feed", cv_display_img)
                            cv2.waitKey(1)

                            if self.needs_water(crop, name):
                                recent_plants.append((self.current_pan, time.time()))

                                print("  Bowing down to water...")

                                # Adjust bow position if the plant is closer (higher tilt value)
                                s0_bow, s2_bow = self.compute_bow()
                                self.set_target(0, s0_bow)
                                self.set_target(2, s2_bow)
                                time.sleep(1.5)

                                print("  *Pretending to water*")
                                for _ in range(100):
                                    w_img = self.capture_image()
                                    cv_w_img = cv2.cvtColor(np.array(w_img), cv2.COLOR_RGB2BGR)
                                    cv2.putText(cv_w_img, "Watering...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (255, 255, 0), 2)
                                    cv2.imshow("Live Feed", cv_w_img)
                                    cv2.waitKey(100)

                                print("  Returning to upright position...")
                                self.set_target(0, SERVO_0_VERTICAL_POS)

                                print("  Returning to scan position for next plant...")
                                self.set_target(1, pan_pos)
                                self.set_target(2, tilt_pos)
                                self.current_pan = pan_pos
                                self.current_tilt = tilt_pos
                                for _ in range(15):
                                    ret_img = self.capture_image()
                                    cv_ret_img = cv2.cvtColor(np.array(ret_img), cv2.COLOR_RGB2BGR)
                                    cv2.putText(cv_ret_img, "Returning...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                    cv2.imshow("Live Feed", cv_ret_img)
                                    cv2.waitKey(100)
                            else:
                                print("  Doesn't need water — skipping.")
                                # Store the estimate so we don't re-identify it either
                                recent_plants.append((estimated_pan, time.time()))
                                # Returning to scan position
                                print("  Returning to scan position for next plant...")
                                self.set_target(1, pan_pos)
                                self.set_target(2, tilt_pos)
                                self.current_pan = pan_pos
                                self.current_tilt = tilt_pos
                                for _ in range(15):
                                    ret_img = self.capture_image()
                                    cv_ret_img = cv2.cvtColor(np.array(ret_img), cv2.COLOR_RGB2BGR)
                                    cv2.putText(cv_ret_img, "Returning...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                    cv2.imshow("Live Feed", cv_ret_img)
                                    cv2.waitKey(100)

        # The loop runs indefinitely until KeyboardInterrupt

        print("\nResetting arm to homed position.")
        self.set_target(0, SERVO_0_VERTICAL_POS)
        self.set_target(1, 6000)
        self.set_target(2, SERVO_2_VERTICAL_POS)
        time.sleep(1.0)

        self.cleanup()
        print("=== Cycle complete — sleeping. ===\n")

    def cleanup(self):
        cv2.destroyAllWindows()
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