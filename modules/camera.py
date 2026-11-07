"""Camera management module for SignBridge.

Handles webcam connection, frame reading with mirroring, status checks,
and graceful error handling for missing/busy cameras.
"""

import cv2
from typing import Optional, Tuple
from utils.config import DEFAULT_CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class Camera:
    """Manages OpenCV webcam capture with friendly error diagnostics."""

    def __init__(self, camera_index: int = DEFAULT_CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False

    def start(self) -> Tuple[bool, str]:
        """Opens the webcam and configures resolution.

        Returns:
            (success, message): True with a success message if opened,
                               False with troubleshooting steps if failed.
        """
        try:
            # On Windows, cv2.CAP_DSHOW can provide faster startup and direct access
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend if DirectShow fails
                self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                self.is_running = False
                return False, (
                    f"Could not open camera index {self.camera_index}. "
                    "Please verify that your webcam is connected, not in use by another app "
                    "(e.g., Zoom/Teams/Browser), and that camera access permissions are enabled in Windows Settings."
                )

            # Set requested dimensions
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.is_running = True
            return True, f"Camera {self.camera_index} connected successfully."

        except Exception as e:
            self.is_running = False
            return False, f"Unexpected error while opening camera: {str(e)}"

    def read_frame(self) -> Tuple[bool, Optional[cv2.typing.MatLike], str]:
        """Reads a single frame from the webcam and applies horizontal mirror.

        Returns:
            (success, frame, status): Frame in BGR format, mirrored horizontally.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None, "Camera is not initialized. Call start() first."

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None, "Failed to capture frame from webcam. Device might have disconnected."

        # Mirror horizontally so signer movements match a mirror view
        frame = cv2.flip(frame, 1)
        return True, frame, "OK"

    def stop(self) -> None:
        """Releases the webcam hardware cleanly."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
