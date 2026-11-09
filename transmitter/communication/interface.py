"""Minimal immutable Tx-to-Rx message contract."""

from dataclasses import dataclass
from typing import Literal

Mode = Literal["natural_language", "json", "vector"]


@dataclass(frozen=True)
class MessageEnvelope:
    experiment_id: str
    image_id: str
    mode: Mode
    payload: str
    encoding_version: str = "1"

    def as_dict(self) -> dict[str, str]:
        return {
            "experiment_id": self.experiment_id,
            "image_id": self.image_id,
            "mode": self.mode,
            "payload": self.payload,
            "encoding_version": self.encoding_version,
        }
