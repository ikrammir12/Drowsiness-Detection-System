"""
src/detector.py
───────────────
Core drowsiness detection engine.

Responsibilities:
  1. Open webcam / video source
  2. Run MediaPipe Face Mesh on each frame
  3. Extract eye landmarks → compute EAR
  4. Count consecutive closed-eye frames
  5. Trigger alerts via AlertManager
  6. Render UI via DisplayRenderer
  7. Log events via SessionLogger
  8. Handle keyboard shortcuts

Algorithm Summary:
─────────────────
  For each frame:
    ┌─ Detect face landmarks (MediaPipe)
    ├─ Extract 6 points per eye
    ├─ Compute EAR for each eye
    ├─ Average left + right EAR
    ├─ If avg_EAR < threshold:
    │     increment counter
    │     if counter >= CONSEC_FRAMES:
    │         ALERT (audio + visual + vibration_sim)
    └─ Else:
          reset counter
          clear alert
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

# ── MediaPipe compatibility shim ──────────────────────────────────
# mediapipe <= 0.10.13  uses:  mp.solutions.face_mesh.FaceMesh
# mediapipe >= 0.10.14  moved: mediapipe.python.solutions.face_mesh
# This shim tries the old path first and falls back automatically.
try:
    import mediapipe as mp
    FaceMesh = mp.solutions.face_mesh.FaceMesh
except AttributeError:
    from mediapipe.python.solutions.face_mesh import FaceMesh
# ─────────────────────────────────────────────────────────────────

from config.settings import Settings
from src.ear_calculator import (
    compute_ear, average_ear, extract_eye_coords,
    get_eye_outline_coords, EARSmoother
)
from src.alert_manager import AlertManager
from src.display import DisplayRenderer
from src.logger import SessionLogger


class DrowsinessDetector:
    """
    Main detection class. Call detector.run() to start the system.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index

        # MediaPipe Face Mesh (compatible with all mediapipe versions)
        self.face_mesh = FaceMesh(
            max_num_faces=1,
            refine_landmarks=Settings.REFINE_LANDMARKS,
            min_detection_confidence=Settings.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Settings.MIN_TRACKING_CONFIDENCE,
        )

        # Sub-modules
        self.alert_manager = AlertManager()
        self.smoother = EARSmoother(window_size=5)
        self.logger = SessionLogger()

        # State
        self.consec_frames = 0          # Consecutive closed-eye frames
        self.frame_counter = 0          # Total frames processed
        self.total_alerts = 0           # Total alert events in session
        self.alert_active = False
        self._alert_just_started = False

        # FPS tracking
        self._fps_counter = 0
        self._fps_timer = time.time()
        self.current_fps = 0.0

        # Display (initialized after we know frame size)
        self.renderer: DisplayRenderer = None

        print("  [Detector] MediaPipe Face Mesh initialized")
        print(f"  [Detector] EAR Threshold = {Settings.EAR_THRESHOLD}")
        print(f"  [Detector] Consec Frames  = {Settings.CONSEC_FRAMES}")

    def run(self):
        """Main detection loop."""
        cap = self._open_camera()
        if cap is None:
            return

        # Get actual frame dimensions
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.renderer = DisplayRenderer(fw, fh)

        print(f"\n  [Detector] Camera opened: {fw}×{fh}")
        print("  [Detector] Detection loop running...\n")

        os.makedirs(Settings.SCREENSHOT_DIR, exist_ok=True)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("  [Detector] Frame read failed — camera disconnected?")
                    break

                self.frame_counter += 1
                self._update_fps()

                # ── Core detection ──
                left_ear, right_ear, avg_ear, face_detected = \
                    self._process_frame(frame)

                # ── State machine ──
                self._update_state(avg_ear, face_detected)

                # ── Render ──
                display_frame = self.renderer.render(
                    frame=frame,
                    left_ear=left_ear,
                    right_ear=right_ear,
                    avg_ear=avg_ear,
                    frame_counter=self.frame_counter,
                    consec_frames=self.consec_frames,
                    alert_active=self.alert_active,
                    vibration_active=self.alert_manager.is_vibrating,
                    vibration_intensity=self.alert_manager.vibration_intensity,
                    vibration_phase=self.alert_manager.vibration_phase,
                    fps=self.current_fps,
                    face_detected=face_detected,
                    total_alerts=self.total_alerts
                )

                cv2.imshow(Settings.WINDOW_TITLE, display_frame)

                # ── Keyboard handling ──
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    print("\n  [Detector] Quit by user")
                    break
                elif key == ord('r'):
                    self._reset_state()
                    print("  [Detector] State reset by user")
                elif key == ord('s'):
                    self._save_screenshot(display_frame)

        except KeyboardInterrupt:
            print("\n  [Detector] Interrupted by user (Ctrl+C)")
        finally:
            self._cleanup(cap)

    # ──────────────────────────────────────────────────────────────
    #  Frame processing
    # ──────────────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray):
        """
        Run MediaPipe and compute EAR.
        Returns (left_ear, right_ear, avg_ear, face_detected).
        """
        h, w = frame.shape[:2]

        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return 0.0, 0.0, 0.0, False

        face_lms = results.multi_face_landmarks[0]

        # Extract 6 key points per eye
        left_coords = extract_eye_coords(
            face_lms, Settings.LEFT_EYE_INDICES, w, h
        )
        right_coords = extract_eye_coords(
            face_lms, Settings.RIGHT_EYE_INDICES, w, h
        )

        # Compute EAR
        left_ear = compute_ear(left_coords)
        right_ear = compute_ear(right_coords)
        raw_avg = average_ear(left_ear, right_ear)
        smooth_avg = self.smoother.smooth(raw_avg)

        # Draw eye outlines on frame
        self._draw_eye_outlines(frame, face_lms, w, h, smooth_avg)

        return left_ear, right_ear, smooth_avg, True

    def _draw_eye_outlines(self, frame, face_lms, w, h, avg_ear):
        """Draw eye landmark outlines on the video frame."""
        color = Settings.COLOR_OK if avg_ear >= Settings.EAR_THRESHOLD \
            else Settings.COLOR_ALERT

        left_outline = get_eye_outline_coords(
            face_lms, Settings.LEFT_EYE_OUTLINE, w, h
        )
        right_outline = get_eye_outline_coords(
            face_lms, Settings.RIGHT_EYE_OUTLINE, w, h
        )

        cv2.polylines(frame, [left_outline], True, color, 1, cv2.LINE_AA)
        cv2.polylines(frame, [right_outline], True, color, 1, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────
    #  State machine
    # ──────────────────────────────────────────────────────────────

    def _update_state(self, avg_ear: float, face_detected: bool):
        """
        Update drowsiness state based on current EAR value.
        """
        if not face_detected:
            # Don't trigger alerts if face is lost (camera moved, etc.)
            # But don't reset counter immediately — wait a few frames
            return

        if avg_ear < Settings.EAR_THRESHOLD:
            self.consec_frames += 1

            if self.consec_frames >= Settings.CONSEC_FRAMES:
                # ── DROWSINESS CONFIRMED ──
                if not self.alert_active:
                    self.alert_active = True
                    self.total_alerts += 1
                    self._alert_just_started = True
                    print(f"\n  ⚠  [Alert #{self.total_alerts}] "
                          f"DROWSINESS at frame {self.frame_counter} "
                          f"| EAR={avg_ear:.3f}")

                self.alert_manager.trigger_alert(self.frame_counter)

                # Log every alert frame
                self.logger.log_event(
                    self.frame_counter, 0.0, 0.0, avg_ear,
                    self.consec_frames, True,
                    self.alert_manager.is_vibrating
                )

        else:
            # Eyes open — reset
            if self.alert_active:
                print(f"  ✔  [Recovery] Driver awake again "
                      f"| frame {self.frame_counter} | EAR={avg_ear:.3f}")
            self.alert_active = False
            self.consec_frames = 0
            self.alert_manager.clear_alert()

    def _reset_state(self):
        """Reset all counters (keyboard shortcut R)."""
        self.consec_frames = 0
        self.alert_active = False
        self.alert_manager.clear_alert()
        self.smoother.reset()

    # ──────────────────────────────────────────────────────────────
    #  Utility
    # ──────────────────────────────────────────────────────────────

    def _open_camera(self):
        """Open the webcam and configure it."""
        print(f"  [Detector] Opening camera index {self.camera_index}...")
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print(f"\n  ✗ ERROR: Could not open camera {self.camera_index}")
            print("    Try: python main.py --camera 1")
            print("    Or:  python main.py --demo  (no webcam needed)")
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Settings.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Settings.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, Settings.TARGET_FPS)
        return cap

    def _update_fps(self):
        self._fps_counter += 1
        if self._fps_counter >= 30:
            now = time.time()
            self.current_fps = self._fps_counter / (now - self._fps_timer)
            self._fps_timer = now
            self._fps_counter = 0

    def _save_screenshot(self, frame: np.ndarray):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(Settings.SCREENSHOT_DIR, f"screenshot_{ts}.png")
        cv2.imwrite(path, frame)
        print(f"  [Screenshot] Saved: {path}")

    def _cleanup(self, cap):
        print("\n  [Detector] Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()
        self.alert_manager.cleanup()
        self.logger.close()
        print("  [Detector] Session ended.\n")