"""
demo/demo_runner.py
───────────────────
Runs the full drowsiness detection pipeline WITHOUT a webcam.

How it works:
  - Generates a synthetic video feed (simulated driver face)
  - Simulates EAR values going through normal → drowsy → alert → recovery
  - All alert systems (audio, visual, vibration sim) fire exactly as
    they would with a real camera feed

This lets you demo the project anywhere — no webcam required.
"""

import cv2
import numpy as np
import time
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import Settings
from src.alert_manager import AlertManager
from src.display import DisplayRenderer
from src.logger import SessionLogger


# ──────────────────────────────────────────────────────────────────
#  Synthetic EAR sequence
# ──────────────────────────────────────────────────────────────────
# This script drives the EAR through a realistic scenario:
#   Phase 1 (0–5s):   Normal driving, EAR ~0.28 (awake)
#   Phase 2 (5–8s):   Getting tired, EAR drops ~0.22 (borderline)
#   Phase 3 (8–12s):  Falling asleep, EAR drops to 0.10 → ALERT
#   Phase 4 (12–15s): Alert wakes driver, EAR recovers to 0.27
#   Phase 5 (15–18s): Brief drowsy dip again
#   Phase 6 (18–25s): Back to normal
#   Repeats...

SCENARIO = [
    # (duration_seconds, ear_value, description)
    (5.0,  0.28, "Normal driving — fully awake"),
    (3.0,  0.22, "Getting tired — EAR dropping"),
    (4.0,  0.09, "Falling asleep — ALERT TRIGGERED"),
    (3.0,  0.27, "Alert fired — driver woke up"),
    (2.0,  0.18, "Brief micro-sleep dip"),
    (1.5,  0.09, "Eyes closing again — ALERT"),
    (2.0,  0.26, "Recovery — awake"),
    (4.0,  0.29, "Normal driving"),
]


def get_simulated_ear(elapsed: float) -> tuple[float, str]:
    """Return (ear_value, description) for elapsed time in scenario."""
    total = sum(d for d, _, _ in SCENARIO)
    t = elapsed % total
    acc = 0.0
    for dur, ear, desc in SCENARIO:
        if t < acc + dur:
            # Smooth transition within phase
            phase_t = (t - acc) / dur
            next_ear = ear  # Could interpolate to next phase
            # Add small noise to simulate real EAR jitter
            noise = np.random.normal(0, 0.005)
            return max(0.0, ear + noise), desc
        acc += dur
    return 0.28, "Normal"


def generate_demo_frame(ear: float, frame_idx: int,
                         fw: int = 640, fh: int = 480) -> np.ndarray:
    """
    Generate a synthetic 'driver's face' frame for the demo.
    The eye openness animates according to the current EAR value.
    """
    # Dark background (simulates car interior at night)
    frame = np.zeros((fh, fw, 3), dtype=np.uint8)
    frame[:] = (20, 18, 25)

    cx, cy = fw // 2, fh // 2

    # ── Face shape ──
    # Head
    cv2.ellipse(frame, (cx, cy - 20), (110, 130), 0, 0, 360, (180, 140, 110), -1)
    # Neck
    cv2.rectangle(frame, (cx - 30, cy + 110), (cx + 30, cy + 170), (180, 140, 110), -1)
    # Hair
    cv2.ellipse(frame, (cx, cy - 60), (110, 90), 0, 180, 360, (40, 30, 20), -1)

    # ── Eyes ──
    # Eye openness: map EAR value to pixel height
    eye_open_h = int(np.interp(ear, [0.0, 0.35], [1, 22]))
    eye_open_h = max(1, eye_open_h)
    eye_w = 32
    eye_y = cy - 15

    for ex in [cx - 45, cx + 45]:
        # Eye whites
        cv2.ellipse(frame, (ex, eye_y), (eye_w // 2, eye_open_h),
                    0, 0, 360, (240, 235, 230), -1)
        # Iris
        if eye_open_h > 4:
            cv2.circle(frame, (ex, eye_y), min(10, eye_open_h - 2),
                       (60, 80, 120), -1)
            cv2.circle(frame, (ex, eye_y), min(6, eye_open_h - 4),
                       (20, 25, 40), -1)
        # Eyelid lines
        cv2.ellipse(frame, (ex, eye_y), (eye_w // 2, eye_open_h),
                    0, 0, 360, (80, 50, 30), 1)

    # ── Nose ──
    cv2.line(frame, (cx, cy + 10), (cx - 12, cy + 35), (140, 100, 75), 2)
    cv2.line(frame, (cx, cy + 10), (cx + 12, cy + 35), (140, 100, 75), 2)

    # ── Mouth — neutral ──
    cv2.ellipse(frame, (cx, cy + 65), (22, 8), 0, 0, 180, (130, 80, 75), 2)

    # ── Dashboard glow at bottom ──
    glow = frame.copy()
    cv2.rectangle(glow, (0, fh - 60), (fw, fh), (0, 30, 60), -1)
    cv2.addWeighted(glow, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"DEMO MODE  |  Simulated EAR: {ear:.3f}",
                (10, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (80, 180, 80), 1, cv2.LINE_AA)

    return frame


def run_demo():
    """Main demo loop."""
    print("  [Demo] Starting demo mode...")
    print("  [Demo] Scenario:")
    for i, (d, e, desc) in enumerate(SCENARIO):
        print(f"         {i+1}. {desc} ({d}s, EAR≈{e})")
    print()

    fw, fh = Settings.FRAME_WIDTH, Settings.FRAME_HEIGHT
    renderer = DisplayRenderer(fw, fh)
    alert_manager = AlertManager()
    logger = SessionLogger()

    consec_frames = 0
    total_alerts = 0
    alert_active = False
    frame_counter = 0
    fps = 30.0
    start_time = time.time()

    os.makedirs(Settings.SCREENSHOT_DIR, exist_ok=True)

    try:
        while True:
            elapsed = time.time() - start_time
            ear, phase_desc = get_simulated_ear(elapsed)

            # Simulate left/right EAR with slight asymmetry
            left_ear = ear + np.random.normal(0, 0.003)
            right_ear = ear + np.random.normal(0, 0.003)
            avg_ear = (left_ear + right_ear) / 2.0

            frame_counter += 1

            # State machine (same as real detector)
            if avg_ear < Settings.EAR_THRESHOLD:
                consec_frames += 1
                if consec_frames >= Settings.CONSEC_FRAMES:
                    if not alert_active:
                        alert_active = True
                        total_alerts += 1
                        print(f"\n  ⚠  [Demo Alert #{total_alerts}] "
                              f"DROWSINESS | EAR={avg_ear:.3f} | {phase_desc}")
                    alert_manager.trigger_alert(frame_counter)
                    logger.log_event(frame_counter, left_ear, right_ear,
                                     avg_ear, consec_frames, True,
                                     alert_manager.is_vibrating)
            else:
                if alert_active:
                    print(f"  ✔  [Demo] Driver awake | EAR={avg_ear:.3f}")
                alert_active = False
                consec_frames = 0
                alert_manager.clear_alert()

            # Generate synthetic frame
            frame = generate_demo_frame(avg_ear, frame_counter, fw, fh)

            # Render dashboard
            display = renderer.render(
                frame=frame,
                left_ear=max(0, left_ear),
                right_ear=max(0, right_ear),
                avg_ear=max(0, avg_ear),
                frame_counter=frame_counter,
                consec_frames=consec_frames,
                alert_active=alert_active,
                vibration_active=alert_manager.is_vibrating,
                vibration_intensity=alert_manager.vibration_intensity,
                vibration_phase=alert_manager.vibration_phase,
                fps=fps,
                face_detected=True,
                total_alerts=total_alerts
            )

            cv2.imshow(Settings.WINDOW_TITLE + " [DEMO]", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n  [Demo] Quit by user")
                break
            elif key == ord('s'):
                ts = int(time.time())
                path = os.path.join(Settings.SCREENSHOT_DIR, f"demo_{ts}.png")
                cv2.imwrite(path, display)
                print(f"  [Demo] Screenshot: {path}")

            # Maintain ~30 FPS
            time.sleep(1.0 / 30)

    except KeyboardInterrupt:
        print("\n  [Demo] Interrupted")
    finally:
        cv2.destroyAllWindows()
        alert_manager.cleanup()
        logger.close()
        print("  [Demo] Demo session ended.")
