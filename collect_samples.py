"""Interactive sample collection tool for SignBridge.

Allows signer to select an ISL sign, preview hand tracking landmarks,
and record labelled normalized landmark samples directly to CSV.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from modules.camera import Camera
from modules.hand_tracking import HandTracker
from modules.preprocessing import extract_feature_vector, get_feature_column_names
from utils.config import (
    DATASET_CSV,
    DATASET_DIR,
    SUPPORTED_SIGNS,
    SIGN_LOOKUP,
    DEFAULT_CAMERA_INDEX,
)


def load_existing_sample_counts() -> dict:
    """Returns a dictionary of sample counts per sign ID from the CSV."""
    counts = {s["id"]: 0 for s in SUPPORTED_SIGNS}
    if DATASET_CSV.exists():
        try:
            df = pd.read_csv(DATASET_CSV)
            if "label" in df.columns:
                existing = df["label"].value_counts().to_dict()
                for k, v in existing.items():
                    if k in counts:
                        counts[k] = v
        except Exception:
            pass
    return counts


def append_sample_to_csv(label: str, num_hands: int, features: np.ndarray) -> None:
    """Appends a single normalized sample row to dataset/isl_landmarks.csv."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    cols = ["label", "timestamp", "num_hands"] + get_feature_column_names()
    row = [label, time.time(), num_hands] + features.tolist()

    file_exists = DATASET_CSV.exists()
    df_row = pd.DataFrame([row], columns=cols)
    df_row.to_csv(DATASET_CSV, mode="a", header=not file_exists, index=False)


def run_sample_collector(camera_index: int = DEFAULT_CAMERA_INDEX):
    """Launches the interactive OpenCV sample recorder."""
    print("=" * 65)
    print("       SignBridge - ISL Landmark Sample Collection Tool        ")
    print("=" * 65)
    print("Controls:")
    print("  [0-9]       : Select sign from the list")
    print("  [SPACE]     : Start recording a burst of samples (30 frames)")
    print("  [C]         : Continuous recording toggle (ON/OFF)")
    print("  [Q] / [ESC] : Save & Exit")
    print("=" * 65)

    camera = Camera(camera_index=camera_index)
    success, msg = camera.start()
    if not success:
        print(f"[Error] {msg}")
        return

    tracker = HandTracker()
    counts = load_existing_sample_counts()

    current_idx = 0
    recording_burst = 0
    burst_total = 30
    continuous_mode = False

    window_name = "SignBridge - Sample Collection [Press Q to Quit]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame, status = camera.read_frame()
            if not ret or frame is None:
                continue

            current_sign = SUPPORTED_SIGNS[current_idx]
            sign_id = current_sign["id"]
            sign_name = current_sign["name"]

            # Process hand landmarks
            results = tracker.process_frame(frame)
            num_hands = tracker.count_hands(results)

            # Draw visual landmarks
            display_frame = tracker.draw_landmarks(frame, results)

            # Check if recording
            is_recording = (recording_burst > 0) or continuous_mode

            if is_recording and num_hands > 0:
                features = extract_feature_vector(results)
                append_sample_to_csv(sign_id, num_hands, features)
                counts[sign_id] += 1
                if recording_burst > 0:
                    recording_burst -= 1

            # Render overlay header
            h, w, _ = display_frame.shape

            # Dark top banner
            cv2.rectangle(display_frame, (0, 0), (w, 85), (20, 20, 30), -1)

            # Title & Current Sign
            cv2.putText(
                display_frame,
                f"Sign [{current_idx + 1}/10]: {sign_name.upper()} ({sign_id})",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

            # Instructions & sample count
            cv2.putText(
                display_frame,
                f"Samples: {counts[sign_id]} | Hands Expected: {current_sign['hands']} | Detected: {num_hands}",
                (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                display_frame,
                f"Tip: {current_sign['description'][:75]}...",
                (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            # Bottom status banner
            cv2.rectangle(display_frame, (0, h - 35), (w, h), (20, 20, 30), -1)

            if is_recording:
                rec_color = (0, 0, 255)
                rec_text = (
                    f"● RECORDING BURST ({burst_total - recording_burst}/{burst_total})"
                    if recording_burst > 0
                    else "● CONTINUOUS RECORDING ACTIVE"
                )
                cv2.circle(display_frame, (w - 20, 45), 10, rec_color, -1)
            else:
                rec_color = (0, 200, 0)
                rec_text = "READY | [SPACE] Record Burst | [C] Continuous | [0-9] Change Sign"

            cv2.putText(
                display_frame,
                rec_text,
                (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                rec_color,
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # 27 is ESC
                break
            elif ord("0") <= key <= ord("9"):
                target_idx = (key - ord("0") - 1) % 10
                current_idx = target_idx
                recording_burst = 0
            elif key == 32:  # SPACEBAR: trigger 30-frame burst
                if num_hands > 0:
                    recording_burst = burst_total
                else:
                    print("[Warning] No hands detected in frame! Show your hand before pressing SPACE.")
            elif key in (ord("c"), ord("C")):
                continuous_mode = not continuous_mode
                recording_burst = 0
                print(f"[SignBridge] Continuous mode: {'ON' if continuous_mode else 'OFF'}")

    finally:
        camera.stop()
        tracker.close()
        cv2.destroyAllWindows()
        print("\n[SignBridge] Sample collection closed.")
        print("Updated Sample Counts:")
        for s in SUPPORTED_SIGNS:
            sid = s["id"]
            print(f"  - {s['name']:12s} ({sid}): {counts.get(sid, 0)} samples")


if __name__ == "__main__":
    run_sample_collector()
