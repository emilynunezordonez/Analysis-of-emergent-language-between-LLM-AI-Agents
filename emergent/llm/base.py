"""Backend-agnostic LLM interface shared by Tx, Rx and the optional judge."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ChatMessage:
    """One chat turn. ``images`` are local file paths shown to a vision model."""

    role: str
    content: str
    images: tuple[Path, ...] = ()


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding parameters. ``temperature == 0`` means greedy decoding."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 256
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "top_p": self.top_p, "max_new_tokens": self.max_new_tokens, "seed": self.seed}


@dataclass(frozen=True)
class Generation:
    """Raw model output plus wall-clock generation latency."""

    text: str
    latency_seconds: float
    output_token_count: int | None = None


class Tokenizer(Protocol):
    """The model's own tokenizer, used for budgets and for logged token ids."""

    name: str
    vocab_size: int | None

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def token_strings(self, ids: Sequence[int]) -> list[str]: ...


class LLMBackend(Protocol):
    """A chat model. Real backends ignore ``task`` and ``hints``.

    ``task`` (``tx_extract``, ``tx_free``, ``rx``, ``judge``) and ``hints`` are
    metadata for the mock backend only; they never reach a real model, which
    sees exclusively the chat messages.
    """

    model_id: str
    tokenizer: Tokenizer

    def generate(
        self,
        messages: Sequence[ChatMessage],
        config: GenerationConfig,
        *,
        task: str,
        hints: Mapping[str, Any] | None = None,
    ) -> Generation: ...


@dataclass
class BackendInfo:
    """Model parameters recorded with every episode."""

    backend: str
    model_id: str
    tokenizer: str
    tokenizer_vocab_size: int | None
    extra: dict[str, Any] = field(default_factory=dict)
