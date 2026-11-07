# SignBridge Models Directory

This directory stores the trained machine learning classifiers and evaluation artifacts for SignBridge.

## Model Details
- **File**: `signbridge_classifier.pkl`
- **Algorithm**: Lightweight Scikit-learn Classifier (Random Forest / MLP Classifier)
- **Input**: Normalized 2D/3D hand landmark coordinates extracted from MediaPipe Hands.
- **Output**: Sign classification label, confidence probability, and top class predictions.
