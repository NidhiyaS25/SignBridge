"""Landmark preprocessing and feature normalization for SignBridge.

Normalizes raw hand coordinates to ensure translation and scale invariance.
Produces a consistent 126-dimensional feature vector supporting both
single-hand and two-hand ISL signs.
"""

from typing import Any, List
import numpy as np


def normalize_single_hand(landmarks) -> List[float]:
    """Normalizes a single hand's 21 landmarks.

    Steps:
    1. Translate coordinates relative to wrist (landmark 0).
    2. Scale coordinates by the maximum Euclidean distance from the wrist.
       This makes the representation invariant to hand size and distance from webcam.

    Args:
        landmarks: List of 21 landmark objects, each with x, y, z attributes.

    Returns:
        List of 63 normalized float values: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20].
    """
    coords = []
    # Compatible with both list-of-landmarks and object with .landmark attribute
    landmark_list = getattr(landmarks, "landmark", landmarks)
    for lm in landmark_list:
        coords.append([lm.x, lm.y, lm.z])

    coords = np.array(coords, dtype=np.float32)  # Shape: (21, 3)

    # Wrist is landmark 0
    wrist = coords[0]

    # Translation: center wrist at origin (0, 0, 0)
    translated = coords - wrist

    # Scale: normalize by maximum distance from wrist to any finger landmark
    distances = np.linalg.norm(translated, axis=1)
    max_dist = np.max(distances)

    if max_dist > 1e-6:
        normalized = translated / max_dist
    else:
        normalized = translated

    return normalized.flatten().tolist()


def extract_feature_vector(results: Any) -> np.ndarray:
    """Extracts a fixed 126-dimensional normalized feature vector from MediaPipe results.

    Layout:
    - Index 0 to 62:   Hand 1 (63 normalized values)
    - Index 63 to 125: Hand 2 (63 normalized values, or zeros if only 1 hand detected)

    Hands are ordered deterministically by horizontal wrist position (left-to-right)
    so hand order is consistent across frames.

    Args:
        results: MediaPipe HandLandmarkerResult (or legacy object).

    Returns:
        1D numpy array of length 126.
    """
    TOTAL_FEATURES = 126  # 2 hands * 21 landmarks * 3 coordinates (x, y, z)

    # Check for hand landmarks in modern and legacy results
    detected_hands = None
    if results is not None:
        if hasattr(results, "hand_landmarks") and results.hand_landmarks:
            detected_hands = results.hand_landmarks
        elif hasattr(results, "multi_hand_landmarks") and results.multi_hand_landmarks:
            detected_hands = results.multi_hand_landmarks

    if not detected_hands:
        return np.zeros(TOTAL_FEATURES, dtype=np.float32)

    # Helper to get wrist x-coordinate
    def get_wrist_x(h):
        lms = getattr(h, "landmark", h)
        return lms[0].x

    # Sort hands left-to-right based on wrist x-coordinate for deterministic ordering
    sorted_hands = sorted(detected_hands, key=get_wrist_x)

    features: List[float] = []

    # Process first hand
    hand1_features = normalize_single_hand(sorted_hands[0])
    features.extend(hand1_features)

    # Process second hand if available, otherwise pad with zeros
    if len(sorted_hands) > 1:
        hand2_features = normalize_single_hand(sorted_hands[1])
        features.extend(hand2_features)
    else:
        # Zero padding for absent second hand
        features.extend([0.0] * 63)

    return np.array(features, dtype=np.float32)


def get_feature_column_names() -> List[str]:
    """Generates column header names for the 126 landmark features."""
    cols = []
    for hand_idx in (1, 2):
        for lm_idx in range(21):
            cols.extend([
                f"h{hand_idx}_lm{lm_idx}_x",
                f"h{hand_idx}_lm{lm_idx}_y",
                f"h{hand_idx}_lm{lm_idx}_z"
            ])
    return cols
