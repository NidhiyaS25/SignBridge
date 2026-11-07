"""Diagnostic script to test webcam access and MediaPipe hand tracking.

Verifies that OpenCV can access the camera and that MediaPipe can process frames.
"""

import sys
import cv2
from modules.camera import Camera
from modules.hand_tracking import HandTracker
from modules.preprocessing import extract_feature_vector


def run_diagnostics():
    print("=" * 60)
    print("       SignBridge - Camera & MediaPipe Diagnostics       ")
    print("=" * 60)

    # 1. OpenCV Version Check
    print(f"[1/4] OpenCV Version: {cv2.__version__}")

    # 2. Camera Connection Test
    print("[2/4] Testing Camera Access (Index 0)...")
    cam = Camera(camera_index=0)
    success, msg = cam.start()

    if not success:
        print(f"[-] FAILED: {msg}")
        return False

    ret, frame, status = cam.read_frame()
    if not ret or frame is None:
        print(f"[-] FAILED: {status}")
        cam.stop()
        return False

    h, w, c = frame.shape
    print(f"[+] Camera is working! Resolution: {w}x{h}, Channels: {c}")

    # 3. MediaPipe Integration Test
    print("[3/4] Initializing MediaPipe Tasks HandLandmarker...")
    try:
        tracker = HandTracker()
        results = tracker.process_frame(frame)
        num_hands = tracker.count_hands(results)
        print(f"[+] MediaPipe successfully processed frame! Hands detected: {num_hands}")

        # Test preprocessing normalization
        features = extract_feature_vector(results)
        print(f"[+] Preprocessing feature vector shape: {features.shape} (Expected: 126)")
        tracker.close()
    except Exception as e:
        print(f"[-] FAILED to run MediaPipe: {e}")
        cam.stop()
        return False

    # 4. Save Test Frame (Verification Artifact)
    test_out = "camera_test.png"
    cv2.imwrite(test_out, frame)
    print(f"[4/4] Saved test capture frame to '{test_out}'")

    cam.stop()
    print("=" * 60)
    print("[SUCCESS] Camera and MediaPipe hand tracking verified!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)
