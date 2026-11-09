"""Consistent per-message metrics with a documented fallback tokenizer."""

import re
from dataclasses import dataclass

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class MessageMetrics:
    character_count: int
    word_count: int
    token_count: int
    tokens: tuple[str, ...]
    tokenizer: str


def tokenize(message: str) -> tuple[str, ...]:
    """Baseline tokenizer: words and punctuation are separate tokens."""
    return tuple(_TOKEN_PATTERN.findall(message))


def measure(message: str) -> MessageMetrics:
    return MessageMetrics(
        character_count=len(message),
        word_count=len(message.split()),
        token_count=len(tokenize(message)),
        tokens=tokenize(message),
        tokenizer="regex-word-punctuation-v1",
    )
