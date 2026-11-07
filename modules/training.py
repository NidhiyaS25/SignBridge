"""Model training and evaluation module for SignBridge.

Loads collected landmark samples from CSV, trains a lightweight Random Forest
classifier, evaluates performance with a train/test split, and serializes
the trained model for real-time inference.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from utils.config import (
    DATASET_CSV,
    MODEL_PATH,
    MODELS_DIR,
    SIGN_IDS,
    SIGN_LOOKUP,
)


def train_sign_model(
    csv_path: Path = DATASET_CSV,
    model_output_path: Path = MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[bool, Dict[str, Any]]:
    """Trains a classifier on collected landmark samples.

    Args:
        csv_path: Path to dataset CSV file.
        model_output_path: Destination path for the serialized model.
        test_size: Fraction of samples reserved for test evaluation.
        random_state: Seed for reproducibility.

    Returns:
        (success, info_dict): Status and training performance summary.
    """
    if not csv_path.exists():
        return False, {
            "error": f"Dataset file not found at {csv_path}. Please collect samples first."
        }

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, {"error": f"Failed to read dataset CSV: {e}"}

    if "label" not in df.columns:
        return False, {"error": "Dataset CSV is missing the required 'label' column."}

    # Extract features (all columns except metadata like label, timestamp, num_hands)
    metadata_cols = {"label", "timestamp", "num_hands"}
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    if len(feature_cols) == 0:
        return False, {"error": "No feature columns found in dataset."}

    X = np.asarray(df[feature_cols].values, dtype=np.float32)
    y = np.asarray(df["label"].values, dtype=str)

    classes = np.unique(y)
    if len(classes) < 2:
        return False, {
            "error": f"At least 2 distinct sign classes are needed to train a classifier. Found: {len(classes)}."
        }

    # Check minimum samples per class for stratified split
    class_counts = df["label"].value_counts()
    min_samples = class_counts.min()

    use_stratify = min_samples >= 2
    stratify = y if use_stratify else None

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Lightweight Random Forest: Fast inference (<2ms), robust to noise, native probabilities
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        random_state=random_state,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Prepare model artifact package
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": clf,
        "classes": list(clf.classes_),
        "feature_names": feature_cols,
        "test_accuracy": float(test_accuracy),
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "trained_at": datetime.now().isoformat(),
        "per_class_counts": class_counts.to_dict(),
    }

    joblib.dump(artifact, model_output_path)

    summary = {
        "accuracy": float(test_accuracy),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "classes": list(clf.classes_),
        "per_class_counts": class_counts.to_dict(),
        "model_path": str(model_output_path),
    }

    return True, summary


if __name__ == "__main__":
    print("[SignBridge] Starting Model Training...")
    ok, res = train_sign_model()
    if ok:
        print(f"[SignBridge] Training Succeeded! Test Accuracy: {res['accuracy']:.2%}")
        print(f"[SignBridge] Classes ({len(res['classes'])}): {res['classes']}")
    else:
        print(f"[SignBridge] Training Failed: {res.get('error')}")
