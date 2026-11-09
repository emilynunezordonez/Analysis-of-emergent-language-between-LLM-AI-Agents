"""Empirical Shannon entropy and frequency distributions."""

from collections import Counter
from math import log2
from typing import Iterable, Hashable


def frequency_distribution(values: Iterable[Hashable]) -> dict[str, int]:
    counts = Counter(values)
    return {str(key): value for key, value in sorted(counts.items(), key=lambda item: str(item[0]))}


def shannon_entropy(values: Iterable[Hashable]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * log2(count / total) for count in counts.values())


def character_entropy(messages: Iterable[str]) -> float:
    return shannon_entropy(character for message in messages for character in message)


def token_entropy(token_sequences: Iterable[Iterable[str]]) -> float:
    return shannon_entropy(token for sequence in token_sequences for token in sequence)
