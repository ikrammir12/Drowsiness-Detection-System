"""
config/settings.py
──────────────────
Central configuration for the Driver Drowsiness Detection System.
Adjust these values to tune detection sensitivity.
"""


class Settings:
    # ──────────────────────────────────────────────────────────────
    # EAR (Eye Aspect Ratio) Detection Parameters
    # ──────────────────────────────────────────────────────────────

    # EAR value below which an eye is considered "closed"
    # Typical range: 0.18 (very sensitive) to 0.25 (less sensitive)
    # Default 0.21 works well for most people in normal lighting
    EAR_THRESHOLD: float = 0.21

    # Number of CONSECUTIVE frames where EAR < threshold to trigger alert
    # At 30 FPS: 20 frames ≈ 0.67 seconds  |  30 frames ≈ 1 second
    # Normal blink = 2–4 frames — we ignore those
    CONSEC_FRAMES: int = 20

    # Minimum EAR for eyes to be considered "open" (resets counter)
    EAR_OPEN_THRESHOLD: float = 0.25

    # ──────────────────────────────────────────────────────────────
    # Camera / Video Settings
    # ──────────────────────────────────────────────────────────────
    TARGET_FPS: int = 30
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480

    # ──────────────────────────────────────────────────────────────
    # MediaPipe Settings
    # ──────────────────────────────────────────────────────────────

    # Minimum detection confidence for face detection
    MIN_DETECTION_CONFIDENCE: float = 0.5

    # Minimum tracking confidence for face tracking
    MIN_TRACKING_CONFIDENCE: float = 0.5

    # Refine landmark detection (gets more accurate eye/iris points)
    REFINE_LANDMARKS: bool = True

    # ──────────────────────────────────────────────────────────────
    # Eye Landmark Indices (MediaPipe Face Mesh — 468 points)
    # These are the 6 key points around each eye used for EAR
    # ──────────────────────────────────────────────────────────────

    # Left eye landmark indices
    LEFT_EYE_INDICES: list = [362, 385, 387, 263, 373, 380]

    # Right eye landmark indices
    RIGHT_EYE_INDICES: list = [33, 160, 158, 133, 153, 144]

    # Full eye outline for visualization
    LEFT_EYE_OUTLINE: list = [
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398
    ]
    RIGHT_EYE_OUTLINE: list = [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246
    ]

    # ──────────────────────────────────────────────────────────────
    # Alert Settings
    # ──────────────────────────────────────────────────────────────
    SOUND_ENABLED: bool = True

    # Alert sound file (relative to project root)
    ALERT_SOUND_FILE: str = "assets/alert.wav"

    # How many times to repeat alert sound while drowsy
    ALERT_REPEAT_INTERVAL: int = 60  # frames between repeats

    # ──────────────────────────────────────────────────────────────
    # Seat Vibration Simulation Settings (DEMO FEATURE)
    # ──────────────────────────────────────────────────────────────
    VIBRATION_ENABLED: bool = True

    # Vibration pattern (milliseconds on, milliseconds off)
    # Simulates: strong buzz — pause — strong buzz — pause — long buzz
    VIBRATION_PATTERN: list = [300, 100, 300, 100, 800]

    # How long each vibration cycle lasts (seconds)
    VIBRATION_DURATION: float = 3.0

    # Vibration intensity levels: 1-10 (affects visual sim)
    VIBRATION_INTENSITY: int = 8

    # ──────────────────────────────────────────────────────────────
    # Display / UI Settings
    # ──────────────────────────────────────────────────────────────

    # Main window title
    WINDOW_TITLE: str = "Driver Drowsiness Detection System | FYP — Subhan"

    # Dashboard panel width (pixels) added beside video
    DASHBOARD_WIDTH: int = 320

    # Colors (BGR format for OpenCV)
    COLOR_OK: tuple = (0, 220, 0)          # Green — awake
    COLOR_WARNING: tuple = (0, 165, 255)   # Orange — borderline
    COLOR_ALERT: tuple = (0, 0, 255)       # Red — drowsy
    COLOR_ACCENT: tuple = (255, 200, 0)    # Cyan-ish accent
    COLOR_WHITE: tuple = (255, 255, 255)
    COLOR_BLACK: tuple = (0, 0, 0)
    COLOR_DARK_BG: tuple = (20, 20, 35)    # Dark panel background
    COLOR_PANEL: tuple = (35, 35, 55)

    # ──────────────────────────────────────────────────────────────
    # Screenshot Settings
    # ──────────────────────────────────────────────────────────────
    SCREENSHOT_DIR: str = "screenshots"

    # ──────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────
    LOG_FILE: str = "logs/drowsiness_log.csv"
    LOG_ENABLED: bool = True
