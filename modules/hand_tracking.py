"""Hand tracking module using MediaPipe Tasks HandLandmarker.

Compatible with Python 3.12+ and modern MediaPipe Tasks API.
Detects 21 3D hand landmarks on one or both hands, draws visualization
overlays with custom styling, and extracts structured landmark arrays.
"""

import os
from pathlib import Path
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

from utils.config import (
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODELS_DIR,
)

# Remote URL for Google's official MediaPipe hand landmarker model
TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
TASK_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"


def ensure_model_asset() -> Path:
    """Ensures the hand_landmarker.task model is available locally."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not TASK_MODEL_PATH.exists() or TASK_MODEL_PATH.stat().st_size < 1000:
        print(f"[SignBridge] Downloading hand_landmarker model to {TASK_MODEL_PATH}...")
        urllib.request.urlretrieve(TASK_MODEL_URL, str(TASK_MODEL_PATH))
        print("[SignBridge] Download complete.")
    return TASK_MODEL_PATH


class HandTracker:
    """Wrapper around MediaPipe Tasks HandLandmarker for robust 21-point tracking."""

    def __init__(
        self,
        max_num_hands: int = MAX_NUM_HANDS,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ):
        model_path = ensure_model_asset()
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.connections = list(HandLandmarksConnections.HAND_CONNECTIONS)

    def process_frame(self, frame: np.ndarray):
        """Processes a BGR OpenCV frame and returns HandLandmarkerResult.

        Args:
            frame: BGR numpy image from webcam.

        Returns:
            HandLandmarkerResult with hand_landmarks and handedness.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect(mp_image)
        return result

    def draw_landmarks(
        self, frame: np.ndarray, results, draw_labels: bool = True
    ) -> np.ndarray:
        """Renders 21 landmarks and bone connections with accessible styling.

        Styling:
        - Connection bones: Vibrant blue / navy lines.
        - Landmark joints: Bright cyan / gold circles.
        - Wrist label: Handedness (Left/Right) with confidence.
        """
        annotated = frame.copy()
        if not results or not results.hand_landmarks:
            return annotated

        h, w, _ = annotated.shape

        for hand_idx, landmarks in enumerate(results.hand_landmarks):
            # Convert normalized coordinates (0..1) to pixel coordinates
            pts = []
            for lm in landmarks:
                px = int(lm.x * w)
                py = int(lm.y * h)
                pts.append((px, py))

            # Draw skeletal connections
            for conn in self.connections:
                start_pt = pts[conn.start]
                end_pt = pts[conn.end]
                cv2.line(annotated, start_pt, end_pt, (235, 120, 20), 2, cv2.LINE_AA)

            # Draw landmark nodes
            for idx, pt in enumerate(pts):
                # Highlight fingertips (landmarks 4, 8, 12, 16, 20) with larger cyan circles
                if idx in (4, 8, 12, 16, 20):
                    cv2.circle(annotated, pt, 5, (0, 220, 255), -1, cv2.LINE_AA)
                    cv2.circle(annotated, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(annotated, pt, 3, (255, 200, 0), -1, cv2.LINE_AA)

            # Draw handedness label near wrist (landmark 0)
            if draw_labels and results.handedness and hand_idx < len(results.handedness):
                cat = results.handedness[hand_idx][0]
                label = f"{cat.category_name} ({cat.score:.0%})"
                wx, wy = pts[0]
                cv2.putText(
                    annotated,
                    label,
                    (wx - 25, max(25, wy - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    label,
                    (wx - 25, max(25, wy - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (30, 60, 220),
                    1,
                    cv2.LINE_AA,
                )

        return annotated

    def count_hands(self, results) -> int:
        """Returns the number of hands detected (0, 1, or 2)."""
        if results and results.hand_landmarks:
            return len(results.hand_landmarks)
        return 0

    def close(self) -> None:
        """Releases the underlying MediaPipe landmarker."""
        if hasattr(self, "detector") and self.detector:
            self.detector.close()
