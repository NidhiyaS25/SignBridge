"""Message builder for SignBridge.

Manages sentence assembly from recognized ISL signs, provides controls to
add words, delete the last word, clear the message, and enforce a cooldown
to prevent accidental duplicate additions from consecutive frames.
"""

import time
from typing import List, Optional
from utils.config import ADD_WORD_COOLDOWN_SEC, SIGN_LOOKUP


class SentenceBuilder:
    """Maintains the active message constructed from recognized sign glosses."""

    def __init__(self, cooldown_seconds: float = ADD_WORD_COOLDOWN_SEC):
        self.words: List[str] = []
        self.cooldown_seconds = cooldown_seconds
        self.last_addition_time = 0.0
        self.last_added_word: Optional[str] = None

    def add_word(self, sign_id: str, force: bool = False) -> bool:
        """Adds a recognized sign to the sentence buffer.

        Args:
            sign_id: The identifier of the sign (e.g. 'HELLO', 'NAMASTE').
            force: If True, bypasses cooldown and duplicate prevention.

        Returns:
            True if the word was successfully added, False if ignored due to cooldown.
        """
        now = time.time()
        elapsed = now - self.last_addition_time

        # Translate sign_id to clean display word
        sign_info = SIGN_LOOKUP.get(sign_id)
        display_word = sign_info["name"] if sign_info else sign_id.replace("_", " ").title()

        if not force:
            # Check cooldown duration
            if elapsed < self.cooldown_seconds:
                return False

            # If the user is holding the exact same sign, require a slightly longer pause
            if display_word == self.last_added_word and elapsed < (self.cooldown_seconds * 1.5):
                return False

        self.words.append(display_word)
        self.last_addition_time = now
        self.last_added_word = display_word
        return True

    def delete_last_word(self) -> Optional[str]:
        """Removes the most recently added word from the message.

        Returns:
            The removed word, or None if the message was already empty.
        """
        if self.words:
            removed = self.words.pop()
            self.last_added_word = self.words[-1] if self.words else None
            return removed
        return None

    def clear(self) -> None:
        """Empties the current message buffer."""
        self.words.clear()
        self.last_added_word = None
        self.last_addition_time = 0.0

    def get_sentence(self) -> str:
        """Returns the full constructed sentence as a formatted string."""
        return " ".join(self.words).strip()

    def is_empty(self) -> bool:
        """Checks if the sentence has no words."""
        return len(self.words) == 0

    def get_word_count(self) -> int:
        """Returns the number of words in the sentence."""
        return len(self.words)
