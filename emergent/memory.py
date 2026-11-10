"""Optional cross-episode memory, the mechanism that lets a convention emerge.

Without memory every episode is independent: Tx and Rx re-derive a protocol
from scratch and messages stay close to the model's default English. With
memory, each agent sees a short window of recent episodes in its prompt, so the
pair can reuse or compress what worked before (in-context adaptation, no weight
updates). Memory is reset at the start of every (condition, seed) game.

Feedback levels:

* ``tx_feedback``: ``none`` (only Tx's own past messages), ``score`` (+ how well
  Rx did), ``reconstruction`` (+ what Rx rebuilt).
* ``rx_feedback``: ``none`` (only its past inputs and outputs), ``score``,
  ``ground_truth`` (+ the true attributes of *past* episodes, as text).
  This is the usual referential-game learning signal. Rx still never sees any
  image, and never sees the current episode's answer.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any

TX_FEEDBACK = ("none", "score", "reconstruction")
RX_FEEDBACK = ("none", "score", "ground_truth")


@dataclass(frozen=True)
class EpisodeRecap:
    """What is remembered about one finished episode."""

    episode_index: int
    tx_message: str
    rx_received: str
    rx_objects: list[dict[str, str]]
    source_objects: list[dict[str, str]]
    score: float | None


def _objects_text(objects: list[dict[str, str]]) -> str:
    return json.dumps(objects, separators=(",", ":"))


class SharedMemory:
    """Sliding window of recaps, rendered differently for Tx and Rx."""

    def __init__(self, enabled: bool = False, window: int = 5, tx_feedback: str = "score", rx_feedback: str = "ground_truth") -> None:
        if tx_feedback not in TX_FEEDBACK or rx_feedback not in RX_FEEDBACK:
            raise ValueError(f"tx_feedback must be in {TX_FEEDBACK}, rx_feedback in {RX_FEEDBACK}")
        self.enabled = enabled
        self.window = window
        self.tx_feedback = tx_feedback
        self.rx_feedback = rx_feedback
        self._items: deque[EpisodeRecap] = deque(maxlen=max(window, 1))
        self.total_added = 0

    def __len__(self) -> int:
        return len(self._items) if self.enabled else 0

    def add(self, recap: EpisodeRecap) -> None:
        if self.enabled:
            self._items.append(recap)
            self.total_added += 1

    def tx_view(self) -> str | None:
        if not self.enabled or not self._items:
            return None
        lines = []
        for item in self._items:
            line = f"- Episode {item.episode_index}: you sent {item.tx_message!r}"
            if self.tx_feedback in ("score", "reconstruction") and item.score is not None:
                line += f"; partner reconstruction score {item.score:.2f}"
            if self.tx_feedback == "reconstruction":
                line += f"; the image was {_objects_text(item.source_objects)}; partner rebuilt {_objects_text(item.rx_objects)}"
            lines.append(line)
        return "\n".join(lines)

    def rx_view(self) -> str | None:
        if not self.enabled or not self._items:
            return None
        lines = []
        for item in self._items:
            line = f"- Episode {item.episode_index}: you received {item.rx_received!r}; you answered {_objects_text(item.rx_objects)}"
            if self.rx_feedback in ("score", "ground_truth") and item.score is not None:
                line += f"; score {item.score:.2f}"
            if self.rx_feedback == "ground_truth":
                line += f"; correct answer was {_objects_text(item.source_objects)}"
            lines.append(line)
        return "\n".join(lines)

    def config(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "window": self.window, "tx_feedback": self.tx_feedback, "rx_feedback": self.rx_feedback}
