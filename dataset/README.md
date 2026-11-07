# SignBridge Dataset Directory

This directory stores the collected Indian Sign Language (ISL) hand landmark samples used to train and evaluate the sign recognition model.

## File Format
- `isl_landmarks.csv`: Each row contains the sign label (`label`), timestamp, hand count, and normalized coordinates $(x_i, y_i, z_i)$ for the 21 MediaPipe hand landmarks (up to 2 hands = 126 landmark features).

## Ethical Data Notice
In compliance with project standards, samples are recorded directly using the SignBridge sample collection tool with reliable ISL visual references. No synthetic or unauthorized third-party datasets are fabricated.
