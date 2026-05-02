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
MODEL_A_PATH    = "yolo26n.pt"
MODEL_B_PATH    = "model_b.tflite"
CLASS_NAMES_PATH= "class_names.txt"

DEDUP_THRESHOLD = 600

SERVO_0_VERTICAL_POS = 8000
SERVO_2_VERTICAL_POS = 4000

TILT_STEPS = [4000, 5000, 6000]

CAMERA_HEIGHT_CM = 22.7
HOSE_HEIGHT_CM   = 5.0

# Calibrated servo conversions
S0_REF_QMS, S0_REF_DEG = 8000, 2.0
S0_QMS_PER_DEG          = 40.7   # decreasing QMS = more forward tilt

S2_REF_QMS, S2_REF_DEG = 4000, 17.0
S2_QMS_PER_DEG          = 42.5   # increasing QMS = more forward tilt


class DIHRobot:

    def __init__(self):
        print("Connecting to Maestro controller...")
        try:
            self.port = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
            print("Connected to controller!")
            self.set_target(0, SERVO_0_VERTICAL_POS)
            self.set_target(1, 6000)
            self.set_target(2, SERVO_2_VERTICAL_POS)
            self.current_pan  = 6000
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
        self.picam2.set_controls({"AwbEnable": False, "AwbMode": 1})
        self.picam2.start()
        time.sleep(2.0)
        print("Ready.\n")

    def set_target(self, channel, target):
        if not self.port:
            return
        lsb = target & 0x7F
        msb = (target >> 7) & 0x7F
        self.port.write(bytes([0x84, channel, lsb, msb]))

    def capture_image(self):
        frame = self.picam2.capture_array()
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        return Image.fromarray(frame).rotate(90, expand=True)

    def detect_pots(self, image):
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
        return True

    def tilt_qms_to_deg(self, qms):
        """Servo 2 QMS → degrees from vertical. 4000=17°, 7100=90°."""
        return S2_REF_DEG + (qms - S2_REF_QMS) / S2_QMS_PER_DEG

    def deg_to_s2_qms(self, deg):
        """Degrees from vertical → servo 2 QMS, clamped to safe range."""
        qms = S2_REF_QMS + int((deg - S2_REF_DEG) * S2_QMS_PER_DEG)
        return max(4000, min(7100, qms))
        
    def estimate_distance(self):
        current_angle_deg = self.tilt_qms_to_deg(self.current_tilt)
        current_angle_deg = max(S2_REF_DEG, min(90.0, current_angle_deg))
        tilt_rad    = math.radians(current_angle_deg)
        return CAMERA_HEIGHT_CM / math.tan(tilt_rad) if tilt_rad > 0 else 999

    def aim(self, center_x, center_y, image_width, image_height):
        """
        Iteratively center the plant in the frame.
        Uses calibrated S2_QMS_PER_DEG for tilt so self.current_tilt
        accurately reflects the physical servo 2 position after convergence.
        """
        cx, cy    = center_x, center_y
        final_img = None
        final_pot = None

        for _ in range(5):
            normalized_x = (cx / image_width)  - 0.5
            normalized_y = (cy / image_height) - 0.5
            angle_x = normalized_x * CAMERA_FOV
            angle_y = normalized_y * CAMERA_FOV_V

            if abs(normalized_x) < 0.05 and abs(normalized_y) < 0.05:
                print("  Target centered.")
                break

            print(f"  Target requires a horizontal shift of {angle_x:.1f}° and vertical shift of {angle_y:.1f}°.")

            # Pan: 25 qms/deg is correct for servo 1
            target_1_qms = self.current_pan - int(angle_x * 25.0)
            # Tilt: use calibrated factor so self.current_tilt stays accurate
            target_2_qms = self.current_tilt + int(angle_y * S2_QMS_PER_DEG)

            target_1_qms = max(0,    min(16000, target_1_qms))
            target_2_qms = max(4000, min(7100,  target_2_qms))

            self.set_target(1, target_1_qms)
            self.set_target(2, target_2_qms)
            self.current_pan  = target_1_qms
            self.current_tilt = target_2_qms

            time.sleep(1.0)

            img  = self.capture_image()
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

            closest   = min(pots, key=lambda p: abs(p["center_x"]/image_width - 0.5) + abs(p["center_y"]/image_height - 0.5))
            cx, cy    = closest["center_x"], closest["center_y"]
            final_img = img
            final_pot = closest

        return final_img, final_pot

    def compute_bow(self):
        """
        Calculate servo positions to point the hose at the plant.

        After aim() converges, self.current_tilt is the QMS value that had the
        camera centred on the plant. The hose exits parallel to the top arm and
        points 90° away from the camera direction, so we add 90° to that angle.

        Servo 0 leans the whole assembly forward for distant plants; servo 2
        compensates so the absolute hose angle stays correct regardless of how
        much servo 0 moves.
        """
        current_angle_deg = self.tilt_qms_to_deg(self.current_tilt)
        current_angle_deg = max(S2_REF_DEG, min(90.0, current_angle_deg))  # clamp to calibrated range

        bow_angle_deg = current_angle_deg + 90.0

        distance_cm = self.estimate_distance()

        print(f"  self.current_tilt={self.current_tilt}  Camera tilt: {current_angle_deg:.1f}°  →  bow angle: {bow_angle_deg:.1f}°")
        print(f"  Estimated plant distance: {distance_cm:.1f} cm")

        # Servo 0: two modes only
        if distance_cm < 30:
            s0_bow      = 8000
            s0_lean_deg = 0.0
        else:
            s0_bow      = 5000
            s0_lean_deg = (8000 - 5000) / S0_QMS_PER_DEG  # ~73.7°

        # Arc correction: tip hose slightly further down so the water arc
        # lands on the plant rather than overshooting it
        arc_correction_deg = math.degrees(math.atan2(HOSE_HEIGHT_CM, distance_cm))

        # Servo 2 works in its own frame (relative to servo 0's arm).
        # Subtract servo 0's lean so the absolute angle stays at bow_angle_deg,
        # then add arc correction to angle the water stream down onto the plant.
        s2_target_deg = bow_angle_deg - s0_lean_deg + arc_correction_deg
        s2_bow        = self.deg_to_s2_qms(s2_target_deg)

        print(f"  S0 lean: {s0_lean_deg:.1f}°  Arc correction: {arc_correction_deg:.1f}°  S2 target: {s2_target_deg:.1f}°")
        print(f"  →  s0_bow={s0_bow}  s2_bow={s2_bow}")

        return s0_bow, s2_bow

    def _return_to_scan(self, pan_pos, tilt_pos, frames=15):
        """Move back to scan position with a live 'Returning' overlay."""
        print("  Returning to scan position for next plant...")
        self.set_target(1, pan_pos)
        self.set_target(2, tilt_pos)
        self.current_pan  = pan_pos
        self.current_tilt = tilt_pos
        for _ in range(frames):
            img    = self.capture_image()
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.putText(cv_img, "Returning...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Live Feed", cv_img)
            cv2.waitKey(100)

    def run_cycle(self):
        print("=== DIH cycle start ===")

        forward_steps = list(range(4000, 7501, 100))
        recent_plants = []  # list of (centred_pan, timestamp)

        while True:
            for i, tilt_pos in enumerate(TILT_STEPS):
                print(f"\n=== Scanning at tilt position {tilt_pos} ===")
                current_pan_steps = forward_steps if i % 2 == 0 else list(reversed(forward_steps))

                for pan_pos in current_pan_steps:
                    print(f"\n--- Scanning at pan {pan_pos}, tilt {tilt_pos} ---")
                    self.set_target(1, pan_pos)
                    self.set_target(2, tilt_pos)
                    self.current_pan  = pan_pos
                    self.current_tilt = tilt_pos

                    time.sleep(0.1)

                    image        = self.capture_image()
                    cv_img       = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    image_width  = image.size[0]
                    image_height = image.size[1]

                    pots = self.detect_pots(image)

                    for pot in pots:
                        x1, y1, x2, y2 = pot["bbox"]
                        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 128), 2)
                        cv2.putText(cv_img, "Pot", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 128), 2)
                    cv2.imshow("Live Feed", cv_img)
                    cv2.waitKey(1)

                    if not pots:
                        print("No plants detected — moving on.")
                        continue

                    print(f"Detected {len(pots)} pot(s).")
                    for j, pot in enumerate(pots):
                        normalized_x  = (pot["center_x"] / image_width) - 0.5
                        angle_x       = normalized_x * CAMERA_FOV
                        estimated_pan = self.current_pan - int(angle_x * 25.0)

                        if any(abs(estimated_pan - p[0]) < DEDUP_THRESHOLD for p in recent_plants):
                            print(f"  [Plant {j+1}] Already handled — skipping.")
                            continue

                        print(f"\n[Plant {j+1}/{len(pots)}]")

                        centered_img, centered_pot = self.aim(pot["center_x"], pot["center_y"], image_width, image_height)

                        if centered_pot is None:
                            print("  Lost plant during aiming — returning to scan position.")
                            self._return_to_scan(pan_pos, tilt_pos)
                            continue
                            
                        distance_cm = self.estimate_distance()
                        if distance_cm > 40:
                            print(f"  Plant too far ({distance_cm:.1f} cm > 40cm) — skipping.")
                            recent_plants.append((self.current_pan, time.time()))
                            self._return_to_scan(pan_pos, tilt_pos)
                            continue

                        x1, y1, x2, y2 = centered_pot["bbox"]
                        padding = 20
                        crop    = centered_img.crop((x1 - padding, y1 - padding, x2 + padding, y2 + padding))

                        name, conf = self.identify_plant(crop)
                        print(f"  Species: {name} ({conf * 100:.1f}%)")

                        display_img    = self.capture_image()
                        cv_display_img = cv2.cvtColor(np.array(display_img), cv2.COLOR_RGB2BGR)

                        if conf < 0.2:
                            print("  Confidence too low (<20%) — skipping.")
                            cv2.putText(cv_display_img, f"{name} {conf*100:.1f}% (LOW)", (x1, max(40, y1+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            cv2.imshow("Live Feed", cv_display_img)
                            cv2.waitKey(1)
                            self._return_to_scan(pan_pos, tilt_pos)
                            continue

                        cv2.putText(cv_display_img, f"{name} {conf*100:.1f}%", (x1, max(40, y1+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.imshow("Live Feed", cv_display_img)
                        cv2.waitKey(1)

                        if self.needs_water(crop, name):
                            recent_plants.append((self.current_pan, time.time()))

                            print("  Bowing down to water...")
                            s0_bow, s2_bow = self.compute_bow()
                            self.set_target(0, s0_bow)
                            self.set_target(2, s2_bow)
                            time.sleep(1.5)

                            print("  *Pretending to water*")
                            for _ in range(100):
                                w_img    = self.capture_image()
                                cv_w_img = cv2.cvtColor(np.array(w_img), cv2.COLOR_RGB2BGR)
                                cv2.putText(cv_w_img, "Watering...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                cv2.imshow("Live Feed", cv_w_img)
                                cv2.waitKey(100)

                            print("  Returning to upright position...")
                            self.set_target(0, SERVO_0_VERTICAL_POS)
                            self._return_to_scan(pan_pos, tilt_pos)
                        else:
                            print("  Doesn't need water — skipping.")
                            recent_plants.append((estimated_pan, time.time()))
                            self._return_to_scan(pan_pos, tilt_pos)

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
