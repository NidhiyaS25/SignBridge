"""Configuration constants and vocabulary definitions for SignBridge."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "signbridge_classifier.pkl"
DATASET_CSV = DATASET_DIR / "isl_landmarks.csv"

# Camera Settings
DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# MediaPipe Hand Tracking Settings
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.6

# Recognition & Prediction Settings
CONFIDENCE_THRESHOLD = 0.65  # Minimum probability required to accept a sign
SMOOTHING_WINDOW_SIZE = 7   # Number of consecutive frames for voting / smoothing
ADD_WORD_COOLDOWN_SEC = 1.2 # Cooldown to avoid rapid duplicate additions in continuous mode

# Supported ISL Signs Vocabulary (8-10 realistic everyday communication aid signs)
# Includes sign identifier, display name, English meaning, and visual description for signer
SUPPORTED_SIGNS = [
    {
        "id": "HELLO",
        "name": "Hello",
        "meaning": "Hello / Greetings",
        "description": "Open hand raised near head/temple with palm facing outward, moving slightly out.",
        "hands": 1
    },
    {
        "id": "NAMASTE",
        "name": "Namaste",
        "meaning": "Namaste / Respectful Greeting",
        "description": "Both palms pressed together flat in front of the chest in traditional greeting.",
        "hands": 2
    },
    {
        "id": "THANK_YOU",
        "name": "Thank You",
        "meaning": "Thank you / Gratitude",
        "description": "Flat hand with fingers touching chin, then extending forward toward the other person.",
        "hands": 1
    },
    {
        "id": "YES",
        "name": "Yes",
        "meaning": "Yes / Affirmation",
        "description": "Fist with thumb up or closed fist nodding downward to indicate agreement.",
        "hands": 1
    },
    {
        "id": "NO",
        "name": "No",
        "meaning": "No / Negation",
        "description": "Index finger extended, waving gently side-to-side or pinching index+middle to thumb.",
        "hands": 1
    },
    {
        "id": "PLEASE",
        "name": "Please",
        "meaning": "Please / Request",
        "description": "Open flat palm gently rubbing in a circular motion over the chest.",
        "hands": 1
    },
    {
        "id": "HELP",
        "name": "Help",
        "meaning": "Help / Need Assistance",
        "description": "One hand closed with thumb upright, resting on the open palm of the other hand.",
        "hands": 2
    },
    {
        "id": "GOOD",
        "name": "Good",
        "meaning": "Good / Well done",
        "description": "Hand forming a clear thumbs-up gesture held firmly in front of body.",
        "hands": 1
    },
    {
        "id": "WATER",
        "name": "Water",
        "meaning": "Water / Thirsty",
        "description": "'W' shape with index, middle, and ring fingers upright tapped gently near the lips.",
        "hands": 1
    },
    {
        "id": "BYE",
        "name": "Goodbye",
        "meaning": "Goodbye / Farewell",
        "description": "Open palm facing forward, waving side-to-side in a parting gesture.",
        "hands": 1
    }
]

# Quick lookup by sign ID
SIGN_LOOKUP = {s["id"]: s for s in SUPPORTED_SIGNS}
SIGN_IDS = [s["id"] for s in SUPPORTED_SIGNS]
