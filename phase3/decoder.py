from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DecoderResult:
    label: str
    confidence: float
    text: str
    stable_progress: float
    empty_progress: float
    emitted: str | None
    finalized: str | None


class TemporalDecoder:
    def __init__(
        self,
        class_names: Iterable[str],
        confidence_threshold: float = 0.8,
        hold_seconds: float = 2.0,
        empty_seconds: float = 5.0,
        release_seconds: float = 0.35,
        smoothing_frames: int = 5,
    ) -> None:
        self.class_names = tuple(class_names)
        if not self.class_names:
            raise ValueError("At least one class is required")
        if not 0 < confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be in (0, 1]")
        if hold_seconds <= 0 or empty_seconds <= 0 or release_seconds < 0 or smoothing_frames <= 0:
            raise ValueError("Invalid decoder timing")
        self.confidence_threshold = confidence_threshold
        self.hold_seconds = hold_seconds
        self.empty_seconds = empty_seconds
        self.release_seconds = release_seconds
        self.blank_labels = {"nothing", "background", "blank", "no_gesture"}
        self.history: deque[np.ndarray] = deque(maxlen=smoothing_frames)
        self.text = ""
        self.candidate: str | None = None
        self.candidate_since: float | None = None
        self.empty_since: float | None = None
        self.release_since: float | None = None
        self.blocked_label: str | None = None

    def clear(self) -> None:
        self.text = ""
        self.candidate = None
        self.candidate_since = None
        self.empty_since = None
        self.release_since = None
        self.blocked_label = None
        self.history.clear()

    def delete_last(self) -> None:
        self.text = self.text[:-1]

    def add_space(self) -> None:
        if self.text and not self.text.endswith(" "):
            self.text += " "

    def _append(self, label: str) -> str:
        normalized = label.casefold()
        if normalized in {"del", "delete", "backspace"}:
            self.delete_last()
            return "delete"
        if normalized in {"space", "_"}:
            self.add_space()
            return "space"
        if len(label) == 1:
            self.text += label.upper()
        else:
            if self.text and not self.text.endswith(" "):
                self.text += " "
            self.text += f"{label} "
        return label

    def _predict(self, probabilities: np.ndarray) -> tuple[str, float]:
        values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if values.size != len(self.class_names):
            raise ValueError("Probability count does not match class names")
        if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
            raise ValueError("Invalid probabilities")
        self.history.append(values / values.sum())
        average = np.mean(np.stack(tuple(self.history)), axis=0)
        index = int(np.argmax(average))
        return self.class_names[index], float(average[index])

    def update(self, probabilities: np.ndarray, timestamp: float) -> DecoderResult:
        label, confidence = self._predict(probabilities)
        confident = confidence >= self.confidence_threshold
        blank = confident and label.casefold() in self.blank_labels
        emitted = None
        finalized = None

        if blank:
            self.candidate = None
            self.candidate_since = None
            if self.empty_since is None:
                self.empty_since = timestamp
            if self.release_since is None:
                self.release_since = timestamp
            if timestamp - self.release_since >= self.release_seconds:
                self.blocked_label = None
            if self.text and timestamp - self.empty_since >= self.empty_seconds:
                finalized = self.text.strip()
                self.text = ""
                self.empty_since = timestamp
        elif confident:
            self.empty_since = None
            self.release_since = None
            if self.blocked_label is not None and label != self.blocked_label:
                self.blocked_label = None
            if label != self.candidate:
                self.candidate = label
                self.candidate_since = timestamp
            elif self.candidate_since is not None and timestamp - self.candidate_since >= self.hold_seconds:
                if label != self.blocked_label:
                    emitted = self._append(label)
                    self.blocked_label = label
                self.candidate_since = timestamp
        else:
            self.candidate = None
            self.candidate_since = None
            self.empty_since = None
            if self.release_since is None:
                self.release_since = timestamp
            if timestamp - self.release_since >= self.release_seconds:
                self.blocked_label = None

        stable_progress = 0.0
        if self.candidate_since is not None:
            stable_progress = min(1.0, max(0.0, (timestamp - self.candidate_since) / self.hold_seconds))
        empty_progress = 0.0
        if self.empty_since is not None:
            empty_progress = min(1.0, max(0.0, (timestamp - self.empty_since) / self.empty_seconds))
        return DecoderResult(label, confidence, self.text, stable_progress, empty_progress, emitted, finalized)
