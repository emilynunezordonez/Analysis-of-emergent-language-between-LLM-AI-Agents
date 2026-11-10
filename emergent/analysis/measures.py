"""Information-theoretic and descriptive measures for a set of messages."""

from __future__ import annotations

import bz2
import gzip
import lzma
import math
import statistics
import zlib
from typing import Any, Sequence

from .estimators import bootstrap_ci, conditional_entropy, unigram_counts, unigram_entropy

COMPRESSORS = {
    "gzip": lambda data: gzip.compress(data, compresslevel=9, mtime=0),
    "zlib": lambda data: zlib.compress(data, 9),
    "bz2": lambda data: bz2.compress(data, 9),
    "lzma": lambda data: lzma.compress(data, preset=9 | lzma.PRESET_EXTREME),
}


def _safe_log2(value: int) -> float:
    return math.log2(value) if value > 1 else float("nan")


def compression_measures(messages: Sequence[str]) -> dict[str, float]:
    """Compressed size of all messages joined by newlines, as an entropy proxy.

    ``bits_per_char`` = 8 * compressed bytes / characters. Container headers
    (about 20 bytes for gzip, about 60 for lzma) dominate on small corpora, so
    ``bits_per_char_net`` subtracts the size of compressing an empty input.
    Compression is still a loose upper bound on entropy for small inputs, so
    compare conditions at similar total sizes.
    """
    raw = "\n".join(messages).encode("utf-8")
    n_chars = max(sum(len(m) for m in messages), 1)
    result: dict[str, float] = {"raw_bytes": float(len(raw))}
    for name, compress in COMPRESSORS.items():
        size = len(compress(raw)) if raw else 0
        overhead = len(compress(b""))
        result[f"{name}_ratio"] = size / len(raw) if raw else float("nan")
        result[f"{name}_bits_per_char"] = 8 * size / n_chars if raw else float("nan")
        result[f"{name}_bits_per_char_net"] = 8 * max(size - overhead, 0) / n_chars if raw else float("nan")
        result[f"{name}_bytes_per_message"] = size / len(messages) if messages else float("nan")
    return result


def describe(messages: Sequence[str], token_sequences: Sequence[Sequence[int]], tokenizer_vocab_size: int | None = None, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """All language measures for one condition. Entropies are in bits per symbol."""
    chars = [list(m) for m in messages]
    tokens = [list(t) for t in token_sequences]
    token_lengths = [len(t) for t in tokens]
    char_lengths = [len(m) for m in messages]
    token_counts = unigram_counts(tokens)
    char_counts = unigram_counts(chars)
    n_tokens, n_chars = sum(token_lengths), sum(char_lengths)
    vocab, alphabet = len(token_counts), len(char_counts)
    row: dict[str, Any] = {
        "n_messages": len(messages),
        "n_tokens": n_tokens,
        "n_chars": n_chars,
        "mean_tokens": statistics.fmean(token_lengths) if tokens else float("nan"),
        "median_tokens": statistics.median(token_lengths) if tokens else float("nan"),
        "std_tokens": statistics.pstdev(token_lengths) if tokens else float("nan"),
        "mean_chars": statistics.fmean(char_lengths) if messages else float("nan"),
        "vocab_size": vocab,
        "type_token_ratio": vocab / n_tokens if n_tokens else float("nan"),
        "char_alphabet_size": alphabet,
        "token_samples_per_type": n_tokens / vocab if vocab else float("nan"),
        "char_samples_per_type": n_chars / alphabet if alphabet else float("nan"),
    }
    for unit, sequences in (("token", tokens), ("char", chars)):
        row[f"H_{unit}_plugin"] = unigram_entropy(sequences, "plugin")
        row[f"H_{unit}_mm"] = unigram_entropy(sequences, "miller_madow")
        row[f"H_{unit}_mm_ci_low"], row[f"H_{unit}_mm_ci_high"], row[f"H_{unit}_mm_boot_bc"] = bootstrap_ci(sequences, unigram_entropy, n_boot, seed)
        row[f"H_{unit}_cond_mm"] = conditional_entropy(sequences, "miller_madow")
        row[f"H_{unit}_cond_mm_ci_low"], row[f"H_{unit}_cond_mm_ci_high"], row[f"H_{unit}_cond_mm_boot_bc"] = bootstrap_ci(sequences, conditional_entropy, n_boot, seed + 1)
    # Observed-alphabet redundancy uses the plug-in H, consistent with the observed K
    # (H_MM can exceed log2 K_observed). The full-vocabulary version uses H_MM.
    row["redundancy_token_observed"] = 1 - row["H_token_plugin"] / _safe_log2(vocab)
    row["redundancy_token_full_vocab"] = 1 - row["H_token_mm"] / _safe_log2(tokenizer_vocab_size) if tokenizer_vocab_size else float("nan")
    row["redundancy_char_observed"] = 1 - row["H_char_plugin"] / _safe_log2(alphabet)
    row["est_bits_per_message_unigram"] = row["H_token_mm"] * row["mean_tokens"]
    bigram_types = len({(a, b) for t in tokens for a, b in zip(["<s>", *t], t)})
    row["token_bigram_samples_per_type"] = n_tokens / bigram_types if bigram_types else float("nan")
    row["undersampled"] = bool(vocab and n_tokens / vocab < 10)
    row.update(compression_measures(messages))
    return row
