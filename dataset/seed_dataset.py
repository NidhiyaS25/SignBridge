"""Seed dataset generator for SignBridge.

Constructs an initial baseline landmark dataset (dataset/isl_landmarks.csv)
using canonical anatomical ISL hand poses with slight biometric variations (jitter).
This allows the classifier to be trained and evaluated immediately upon installation,
while remaining fully extensible via collect_samples.py.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path so modules can be imported directly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from modules.preprocessing import get_feature_column_names, normalize_single_hand
from utils.config import DATASET_CSV, DATASET_DIR, SUPPORTED_SIGNS


class MockLandmark:
    """Mock landmark point simulating MediaPipe landmark object."""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class MockHand:
    """Mock hand simulating MediaPipe hand landmark list."""
    def __init__(self, points):
        self.landmark = [MockLandmark(x, y, z) for x, y, z in points]


def get_base_open_hand():
    """Generates standard open hand coordinates (palm facing camera)."""
    return [
        [0.50, 0.80, 0.00],  # 0: Wrist
        [0.42, 0.72, -0.02], # 1: Thumb CMC
        [0.36, 0.64, -0.04], # 2: Thumb MCP
        [0.32, 0.56, -0.05], # 3: Thumb IP
        [0.28, 0.50, -0.06], # 4: Thumb TIP
        [0.44, 0.58, -0.02], # 5: Index MCP
        [0.42, 0.46, -0.03], # 6: Index PIP
        [0.41, 0.36, -0.04], # 7: Index DIP
        [0.40, 0.28, -0.04], # 8: Index TIP
        [0.50, 0.56, -0.02], # 9: Middle MCP
        [0.50, 0.42, -0.03], # 10: Middle PIP
        [0.50, 0.32, -0.04], # 11: Middle DIP
        [0.50, 0.24, -0.04], # 12: Middle TIP
        [0.56, 0.58, -0.02], # 13: Ring MCP
        [0.57, 0.45, -0.03], # 14: Ring PIP
        [0.58, 0.35, -0.04], # 15: Ring DIP
        [0.58, 0.27, -0.04], # 16: Ring TIP
        [0.62, 0.62, -0.02], # 17: Pinky MCP
        [0.64, 0.52, -0.03], # 18: Pinky PIP
        [0.66, 0.44, -0.04], # 19: Pinky DIP
        [0.67, 0.37, -0.04], # 20: Pinky TIP
    ]


def get_base_fist(thumb_up=False):
    """Generates closed fist coordinates, optionally with thumb upright."""
    points = get_base_open_hand()
    # Curl index, middle, ring, pinky towards palm
    for base_idx in [5, 9, 13, 17]:
        mcp_y = points[base_idx][1]
        points[base_idx + 1][1] = mcp_y + 0.06  # PIP curled down
        points[base_idx + 2][1] = mcp_y + 0.08  # DIP curled in
        points[base_idx + 3][1] = mcp_y + 0.05  # TIP tucked in

    if thumb_up:
        # Thumb extended upright
        points[1] = [0.44, 0.70, -0.02]
        points[2] = [0.42, 0.58, -0.04]
        points[3] = [0.40, 0.46, -0.06]
        points[4] = [0.38, 0.34, -0.08]
    else:
        # Thumb folded over curled fingers
        points[1] = [0.44, 0.72, -0.02]
        points[2] = [0.46, 0.65, -0.04]
        points[3] = [0.48, 0.60, -0.05]
        points[4] = [0.50, 0.58, -0.06]

    return points


def get_sign_pose(sign_id: str):
    """Returns (hand1_points, hand2_points_or_none) for canonical ISL signs."""
    if sign_id == "HELLO":
        # Open hand raised near temple, fingers spread and angled
        h1 = get_base_open_hand()
        for p in h1:
            p[0] += 0.05
            p[1] -= 0.15  # Raised near head
        return h1, None

    elif sign_id == "NAMASTE":
        # Two flat hands pressed together in center
        h1 = get_base_open_hand()
        for p in h1:
            p[0] = 0.45 + (p[0] - 0.50) * 0.4  # compressed vertically together
        h2 = get_base_open_hand()
        for p in h2:
            p[0] = 0.55 - (p[0] - 0.50) * 0.4  # facing opposite
        return h1, h2

    elif sign_id == "THANK_YOU":
        # Flat hand fingers pointing forward/outward from chin
        h1 = get_base_open_hand()
        for idx, p in enumerate(h1):
            if idx > 0:
                p[1] += 0.08  # angled outward towards listener
                p[2] -= 0.15  # depth forward
        return h1, None

    elif sign_id == "YES":
        # Closed fist nodding downwards
        h1 = get_base_fist(thumb_up=False)
        for p in h1:
            p[1] += 0.05  # lower wrist nod
        return h1, None

    elif sign_id == "NO":
        # Index finger upright, other fingers curled tightly
        h1 = get_base_fist(thumb_up=False)
        h1[6] = [0.44, 0.46, -0.03]
        h1[7] = [0.44, 0.34, -0.04]
        h1[8] = [0.44, 0.22, -0.05]  # Index fully upright
        return h1, None

    elif sign_id == "PLEASE":
        # Open palm flat facing chest
        h1 = get_base_open_hand()
        for p in h1:
            p[2] += 0.10  # pushed against chest
            p[0] -= 0.05
        return h1, None

    elif sign_id == "HELP":
        # Two hands: Hand 1 flat base palm, Hand 2 thumbs-up fist resting on it
        h1 = get_base_open_hand()
        for p in h1:
            p[1] = 0.75 + (p[1] - 0.75) * 0.2  # horizontal flat base
            p[0] = 0.40 + (p[0] - 0.50)
        h2 = get_base_fist(thumb_up=True)
        for p in h2:
            p[0] = 0.55 + (p[0] - 0.50)
            p[1] -= 0.12
        return h1, h2

    elif sign_id == "GOOD":
        # Pure upright thumbs-up gesture
        h1 = get_base_fist(thumb_up=True)
        return h1, None

    elif sign_id == "WATER":
        # 'W' sign: Index, middle, ring extended; thumb holding pinky
        h1 = get_base_open_hand()
        h1[18][1] = h1[17][1] + 0.07
        h1[19][1] = h1[17][1] + 0.09
        h1[20][1] = h1[17][1] + 0.06
        h1[3] = [0.55, 0.60, -0.04]
        h1[4] = [0.60, 0.58, -0.05]
        return h1, None

    elif sign_id == "BYE":
        # Open hand waving with prominent horizontal lateral tilt
        h1 = get_base_open_hand()
        for idx, p in enumerate(h1):
            p[0] += (p[1] - 0.8) * 0.45  # strong wave tilt
        return h1, None

    return get_base_open_hand(), None


def generate_seed_dataset(samples_per_sign: int = 60):
    """Generates a high-quality baseline dataset with biometric variations."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    cols = ["label", "timestamp", "num_hands"] + get_feature_column_names()

    np.random.seed(42)

    for sign in SUPPORTED_SIGNS:
        sign_id = sign["id"]
        h1_base, h2_base = get_sign_pose(sign_id)
        num_hands = 2 if h2_base is not None else 1

        for _ in range(samples_per_sign):
            # Apply slight realistic spatial jitter (+-2% variation)
            jitter_scale = 0.015
            h1_jittered = np.array(h1_base) + np.random.normal(0, jitter_scale, size=np.array(h1_base).shape)
            h1_norm = normalize_single_hand(MockHand(h1_jittered.tolist()))

            if h2_base is not None:
                h2_jittered = np.array(h2_base) + np.random.normal(0, jitter_scale, size=np.array(h2_base).shape)
                h2_norm = normalize_single_hand(MockHand(h2_jittered.tolist()))
            else:
                h2_norm = [0.0] * 63

            full_features = h1_norm + h2_norm
            row = [sign_id, time.time(), num_hands] + full_features
            rows.append(row)

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(DATASET_CSV, index=False)
    print(f"[SignBridge] Successfully generated baseline dataset at {DATASET_CSV}!")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Signs ({len(SUPPORTED_SIGNS)}): {[s['id'] for s in SUPPORTED_SIGNS]}")
    print(f"  - Samples per sign: {samples_per_sign}")


if __name__ == "__main__":
    generate_seed_dataset()
