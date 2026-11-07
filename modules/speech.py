"""Text-to-speech module for SignBridge.

Supports both Python desktop audio (via pyttsx3/SAPI5 on Windows)
and browser-compatible Web Speech API synthesis for Streamlit.
Handles empty message states cleanly.
"""

import tempfile
import html
from pathlib import Path
from typing import Tuple, Optional


class SpeechEngine:
    """Manages audio narration of recognized messages."""

    def __init__(self, rate: int = 150, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self._engine = None

    def _get_engine(self):
        """Lazily initializes the pyttsx3 TTS engine to conserve resources."""
        if self._engine is None:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self.rate)
                self._engine.setProperty("volume", self.volume)
            except Exception as e:
                self._engine = None
        return self._engine

    def speak(self, text: str) -> Tuple[bool, str]:
        """Speaks the given text out loud using the local system speaker.

        Args:
            text: Sentence or word to narrate.

        Returns:
            (success, message): Status of speech operation.
        """
        clean_text = text.strip() if text else ""
        if not clean_text:
            return False, "Cannot speak an empty message. Please sign a word first."

        engine = self._get_engine()
        if engine is None:
            return False, "Speech synthesizer engine is not available on this system."

        try:
            engine.say(clean_text)
            engine.runAndWait()
            return True, f"Narrated: '{clean_text}'"
        except Exception as e:
            return False, f"Speech synthesis error: {str(e)}"

    def generate_audio_file(self, text: str) -> Tuple[bool, Optional[str], str]:
        """Saves synthesized speech to a temporary WAV file for browser playback.

        Args:
            text: Text to synthesize.

        Returns:
            (success, file_path, status_message)
        """
        clean_text = text.strip() if text else ""
        if not clean_text:
            return False, None, "Message is empty."

        engine = self._get_engine()
        if engine is None:
            return False, None, "TTS engine unavailable."

        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_path = temp_file.name
            temp_file.close()

            engine.save_to_file(clean_text, temp_path)
            engine.runAndWait()
            return True, temp_path, "Audio generated successfully."
        except Exception as e:
            return False, None, f"Failed to generate audio file: {str(e)}"

    @staticmethod
    def get_browser_speech_html(text: str) -> str:
        """Generates a client-side Web Speech API JavaScript snippet.

        This guarantees audio playback in any modern web browser without server latency.
        """
        clean_text = text.strip() if text else ""
        if not clean_text:
            return ""

        escaped = html.escape(clean_text)
        return f"""
        <script>
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel(); // Stop any pending speech
                const utterance = new SpeechSynthesisUtterance({repr(clean_text)});
                utterance.rate = 0.95;
                utterance.pitch = 1.0;
                utterance.lang = 'en-IN'; // Indian English if available
                window.speechSynthesis.speak(utterance);
            }}
        </script>
        """
