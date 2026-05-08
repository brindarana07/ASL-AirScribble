import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional, Set


@dataclass
class BuilderEvent:
    kind: str
    value: str = ""


class WordBuilder:
    def __init__(
        self,
        stable_frame_window: int = 15,
        hold_seconds: float = 1.5,
        cooldown_seconds: float = 0.5,
        space_labels: Optional[Set[str]] = None,
        delete_labels: Optional[Set[str]] = None,
        nothing_labels: Optional[Set[str]] = None,
    ):
        self.prediction_buffer: Deque[str] = deque(maxlen=stable_frame_window)
        self.hold_seconds = hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.space_labels = {label.upper() for label in (space_labels or {"SPACE"})}
        self.delete_labels = {label.upper() for label in (delete_labels or {"DELETE"})}
        self.nothing_labels = {label.upper() for label in (nothing_labels or {"IDLE", "NOTHING"})}
        self.current_word = ""
        self.sentence = ""
        self.stable_label = "IDLE"
        self._candidate_label = "IDLE"
        self._candidate_started_at = time.time()
        self._last_confirmed_at = 0.0

    def update(self, predicted_label: str) -> Optional[BuilderEvent]:
        now = time.time()
        label = (predicted_label or "IDLE").upper()
        self.prediction_buffer.append(label)
        majority_label = self._majority_label()
        self.stable_label = majority_label

        if majority_label in self.nothing_labels:
            self._candidate_label = majority_label
            self._candidate_started_at = now
            return None

        if majority_label != self._candidate_label:
            self._candidate_label = majority_label
            self._candidate_started_at = now
            return None

        held_long_enough = now - self._candidate_started_at >= self.hold_seconds
        cooldown_done = now - self._last_confirmed_at >= self.cooldown_seconds
        if not held_long_enough or not cooldown_done:
            return None

        self._last_confirmed_at = now
        self._candidate_started_at = now
        return self._confirm_label(majority_label)

    def add_space(self, correction: Optional[str] = None) -> BuilderEvent:
        if self.current_word:
            word = (correction or self.current_word).upper()
            if self.sentence and not self.sentence.endswith(" "):
                self.sentence += " "
            self.sentence += word
            self.current_word = ""
            return BuilderEvent(kind="word", value=word)
        if self.sentence and not self.sentence.endswith(" "):
            self.sentence += " "
        return BuilderEvent(kind="space", value=" ")

    def backspace(self) -> BuilderEvent:
        if self.current_word:
            removed = self.current_word[-1]
            self.current_word = self.current_word[:-1]
            return BuilderEvent(kind="delete", value=removed)
        if self.sentence:
            removed = self.sentence[-1]
            self.sentence = self.sentence[:-1]
            return BuilderEvent(kind="delete", value=removed)
        return BuilderEvent(kind="delete", value="")

    def clear(self) -> BuilderEvent:
        self.current_word = ""
        self.sentence = ""
        self.prediction_buffer.clear()
        return BuilderEvent(kind="clear", value="")

    def accept_suggestion(self, suggestion: str) -> BuilderEvent:
        if not suggestion:
            return BuilderEvent(kind="ignored", value="")
        self.current_word = suggestion.upper()
        return BuilderEvent(kind="suggestion", value=self.current_word)

    def full_text(self) -> str:
        if self.current_word:
            if self.sentence and not self.sentence.endswith(" "):
                return f"{self.sentence} {self.current_word}"
            return f"{self.sentence}{self.current_word}"
        return self.sentence

    def hold_progress(self) -> float:
        if self._candidate_label in self.nothing_labels:
            return 0.0
        elapsed = time.time() - self._candidate_started_at
        if self.hold_seconds <= 0:
            return 1.0
        return max(0.0, min(1.0, elapsed / self.hold_seconds))

    def _confirm_label(self, label: str) -> BuilderEvent:
        if label in self.space_labels:
            return self.add_space()
        if label in self.delete_labels:
            return self.backspace()
        if len(label) == 1 and label.isalpha():
            self.current_word += label
            return BuilderEvent(kind="letter", value=label)
        return BuilderEvent(kind="ignored", value=label)

    def _majority_label(self) -> str:
        if not self.prediction_buffer:
            return "IDLE"
        return Counter(self.prediction_buffer).most_common(1)[0][0]
