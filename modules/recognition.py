"""ISL sign recognition and classification module for SignBridge.

Uses a trained lightweight scikit-learn model, confidence thresholds,
and temporal prediction smoothing to classify hand landmarks in real time.
"""

from collections import deque, Counter
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import joblib
import numpy as np

from utils.config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    SMOOTHING_WINDOW_SIZE,
    SIGN_LOOKUP,
    SIGN_IDS,
)


class SignClassifier:
    """Predicts ISL signs from normalized hand landmark feature vectors."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        smoothing_window: int = SMOOTHING_WINDOW_SIZE,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.model: Optional[Any] = None
        self.classes: List[str] = []
        self.prediction_history: deque = deque(maxlen=smoothing_window)
        self.load_model()

    def load_model(self) -> bool:
        """Loads the trained model from disk if it exists."""
        if self.model_path.exists():
            try:
                data = joblib.load(self.model_path)
                # Supports both bare model and dict with metadata
                if isinstance(data, dict) and "model" in data:
                    self.model = data["model"]
                    self.classes = list(data.get("classes", self.model.classes_))
                else:
                    self.model = data
                    self.classes = list(self.model.classes_)
                return True
            except Exception as e:
                print(f"[SignBridge] Warning: Could not load model from {self.model_path}: {e}")
                self.model = None
                return False
        return False

    def is_trained(self) -> bool:
        """Checks if a trained model is ready for inference."""
        return self.model is not None

    def predict(
        self, feature_vector: np.ndarray, num_hands_detected: int
    ) -> Dict[str, Any]:
        """Classifies a single frame's 126-dimensional normalized landmarks.

        Args:
            feature_vector: 1D array of shape (126,).
            num_hands_detected: Number of hands found in current frame.

        Returns:
            Dictionary containing:
                - 'sign_id': Predicted sign ID or None if below threshold / no hand.
                - 'name': Display name (e.g. 'Hello').
                - 'meaning': Meaning explanation.
                - 'confidence': Probability score between 0.0 and 1.0.
                - 'raw_sign_id': Unsmoothed prediction.
                - 'is_confident': Boolean flag indicating if threshold was met.
                - 'probabilities': Dict of top class probabilities.
        """
        result: Dict[str, Any] = {
            "sign_id": None,
            "name": "No Hand Detected" if num_hands_detected == 0 else "Analyzing...",
            "meaning": "",
            "confidence": 0.0,
            "raw_sign_id": None,
            "is_confident": False,
            "probabilities": {},
        }

        # If no hands are visible, reset smoothing buffer
        if num_hands_detected == 0 or not self.is_trained():
            self.prediction_history.clear()
            if not self.is_trained():
                result["name"] = "Model Not Trained Yet"
                result["meaning"] = "Please record samples and train model in Dataset tab."
            return result

        # Ensure correct input shape (1, 126)
        x = feature_vector.reshape(1, -1)

        try:
            # Predict class probabilities
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(x)[0]
                best_idx = int(np.argmax(probs))
                best_confidence = float(probs[best_idx])
                raw_pred = str(self.model.classes_[best_idx])

                # Build dictionary of class probabilities for transparency
                for idx, cls_name in enumerate(self.model.classes_):
                    result["probabilities"][str(cls_name)] = float(probs[idx])
            else:
                raw_pred = str(self.model.predict(x)[0])
                best_confidence = 1.0

            result["raw_sign_id"] = raw_pred
            result["confidence"] = best_confidence

            # Enforce confidence threshold
            if best_confidence >= self.confidence_threshold:
                self.prediction_history.append(raw_pred)
            else:
                self.prediction_history.append(None)

            # Apply temporal voting smoothing over the recent sliding window
            valid_predictions = [p for p in self.prediction_history if p is not None]
            if len(valid_predictions) >= (self.smoothing_window // 2 + 1):
                most_common, count = Counter(valid_predictions).most_common(1)[0]
                # Require majority in current buffer
                if count >= (self.smoothing_window // 2):
                    result["sign_id"] = most_common
                    result["is_confident"] = True
                    sign_info = SIGN_LOOKUP.get(most_common, {})
                    result["name"] = sign_info.get("name", most_common)
                    result["meaning"] = sign_info.get("meaning", "")
                else:
                    result["name"] = "Stabilizing..."
            else:
                result["name"] = "Uncertain / Low Confidence"

        except Exception as e:
            result["name"] = f"Inference Error: {e}"

        return result

    def reset_smoothing(self) -> None:
        """Clears the temporal voting queue."""
        self.prediction_history.clear()
