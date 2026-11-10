"""LLM backends. ``build_backend`` creates the single instance shared by Tx and Rx."""

from __future__ import annotations

from typing import Any, Mapping

from .base import BackendInfo, ChatMessage, Generation, GenerationConfig, LLMBackend, Tokenizer
from .mock import MockBackend, MockTokenizer

__all__ = ["BackendInfo", "ChatMessage", "Generation", "GenerationConfig", "LLMBackend", "Tokenizer", "MockBackend", "build_backend", "build_tokenizer", "backend_info"]


def build_backend(cfg: Mapping[str, Any]) -> LLMBackend:
    """Create a backend from the ``backend`` config section (``type: mock | qwen``)."""
    kind = cfg.get("type", "mock")
    if kind == "mock":
        return MockBackend(malformed_rate=float(cfg.get("mock", {}).get("malformed_rate", 0.0)))
    if kind == "qwen":
        from .qwen import QwenBackend

        return QwenBackend(cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct"), cfg.get("device_map", "auto"), cfg.get("torch_dtype", "auto"))
    raise ValueError(f"unknown backend type {kind!r}")


def build_tokenizer(cfg: Mapping[str, Any]) -> Tokenizer:
    """Tokenizer only (no weights), used by the analysis scripts for baselines."""
    if cfg.get("type", "mock") == "mock":
        return MockTokenizer()
    from .qwen import HFTokenizer

    return HFTokenizer.from_pretrained(cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct"))


def backend_info(cfg: Mapping[str, Any], backend: LLMBackend) -> BackendInfo:
    return BackendInfo(
        backend=cfg.get("type", "mock"),
        model_id=backend.model_id,
        tokenizer=backend.tokenizer.name,
        tokenizer_vocab_size=backend.tokenizer.vocab_size,
        extra={k: v for k, v in cfg.items() if k not in {"type", "model_id"}},
    )
