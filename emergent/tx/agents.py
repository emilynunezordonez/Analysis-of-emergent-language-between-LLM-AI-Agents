"""Transmitter adapters. The original ``transmitter`` package is reused, not rewritten.

* ``TemplateTx``: image -> attributes (vision model via the shared backend with
  the existing ``transmitter/prompts/visual_extraction.txt`` prompt, or the
  existing ground-truth ``MockExtractor``) -> one of the existing deterministic
  encoders (``natural_language``, ``json``, ``vector``). These are fixed
  reference schemes and cannot adapt.
* ``LLMFreeTx`` (new mode ``llm_free``): the LLM looks at the image and writes
  its own message. It is optionally told the token budget and shown the
  episode memory. This is the mode in which a convention can emerge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from transmitter.communication import json_format, natural_language, vector_format
from transmitter.dataset.schema import ImageAttributes
from transmitter.model.adapter import MockExtractor

from emergent.llm.base import ChatMessage, GenerationConfig, LLMBackend
from emergent.memory import SharedMemory
from emergent.rx.parsing import extract_json_object

TEMPLATE_MODES = ("natural_language", "json", "vector")
TX_MODES = ("llm_free", *TEMPLATE_MODES)
_EXTRACTION_PROMPT = Path(__file__).resolve().parents[2] / "transmitter" / "prompts" / "visual_extraction.txt"
_FREE_SYSTEM_PROMPT = Path(__file__).with_name("prompts") / "free_form_system.txt"

_ENCODERS = {
    "natural_language": natural_language.encode,
    "json": json_format.encode,
    "vector": lambda attrs: json.dumps(vector_format.encode(attrs), separators=(",", ":")),
}


@dataclass
class TxOutput:
    """Everything Tx produced for one episode. ``message`` is sent unchanged to the channel."""

    mode: str
    message: str
    raw_outputs: list[str]
    latency_seconds: float
    extracted_objects: list[dict[str, str]] | None = None
    errors: list[str] = field(default_factory=list)
    prompt: str | None = None


class TxAgent(Protocol):
    mode: str

    def transmit(self, image_path: Path, ground_truth: ImageAttributes, *, token_budget: int | None, memory: SharedMemory, generation: GenerationConfig) -> TxOutput: ...


class TemplateTx:
    """Existing Tx pipeline: extract attributes, then apply a fixed template encoder.

    ``ground_truth`` is used only when ``extractor == "ground_truth"`` (the
    existing ``MockExtractor`` baseline). The backend extractor sees only the image.
    """

    def __init__(self, mode: str, backend: LLMBackend, extractor: str = "backend", max_retries: int = 2) -> None:
        if mode not in TEMPLATE_MODES:
            raise ValueError(f"template mode must be one of {TEMPLATE_MODES}")
        if extractor not in ("backend", "ground_truth"):
            raise ValueError("extractor must be 'backend' or 'ground_truth'")
        self.mode, self.backend, self.extractor, self.max_retries = mode, backend, extractor, max_retries

    def _extract(self, image_path: Path, ground_truth: ImageAttributes, generation: GenerationConfig) -> tuple[ImageAttributes | None, list[str], list[str], float]:
        if self.extractor == "ground_truth":
            attributes, seconds = MockExtractor().extract(image_path, ground_truth)
            return attributes, [], [], seconds
        prompt = _EXTRACTION_PROMPT.read_text(encoding="utf-8")
        raws, errors, total = [], [], 0.0
        for attempt in range(self.max_retries + 1):
            config = GenerationConfig(generation.temperature, generation.top_p, generation.max_new_tokens, None if generation.seed is None else generation.seed + attempt)
            result = self.backend.generate([ChatMessage("user", prompt, (image_path,))], config, task="tx_extract")
            raws.append(result.text)
            total += result.latency_seconds
            try:
                data = json.loads(extract_json_object(result.text) or "")
                data["image_id"] = image_path.stem  # the model cannot know the file id
                return ImageAttributes.from_dict(data), raws, errors, total
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as error:
                errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
        return None, raws, errors, total

    def transmit(self, image_path: Path, ground_truth: ImageAttributes, *, token_budget: int | None, memory: SharedMemory, generation: GenerationConfig) -> TxOutput:
        del token_budget, memory  # fixed templates cannot adapt; the channel enforces the budget
        attributes, raws, errors, seconds = self._extract(image_path, ground_truth, generation)
        if attributes is None:
            return TxOutput(self.mode, "", raws, seconds, None, errors + ["extraction failed; empty message sent"])
        return TxOutput(self.mode, _ENCODERS[self.mode](attributes), raws, seconds, [o.to_dict() for o in attributes.objects], errors)


class LLMFreeTx:
    """The LLM writes the message itself (mode ``llm_free``)."""

    mode = "llm_free"

    def __init__(self, backend: LLMBackend, inform_budget: bool = True, system_prompt_path: Path = _FREE_SYSTEM_PROMPT) -> None:
        self.backend = backend
        self.inform_budget = inform_budget
        self.system_prompt = system_prompt_path.read_text(encoding="utf-8")

    def build_user_prompt(self, token_budget: int | None, memory: SharedMemory) -> str:
        parts = ["Write your message for the receiver."]
        if token_budget is not None and self.inform_budget:
            parts.append(f"The channel only carries the first {token_budget} tokens of your message; anything after that is cut off.")
        history = memory.tx_view()
        if history:
            parts.append("Recent episodes with the same partner (you may reuse or shorten conventions that worked):\n" + history)
        return "\n\n".join(parts)

    def transmit(self, image_path: Path, ground_truth: ImageAttributes, *, token_budget: int | None, memory: SharedMemory, generation: GenerationConfig) -> TxOutput:
        del ground_truth  # never used: the real model only sees the image
        user_prompt = self.build_user_prompt(token_budget, memory)
        hints: dict[str, Any] = {"token_budget": token_budget if self.inform_budget else None, "history_len": memory.total_added if memory.enabled else 0}
        result = self.backend.generate([ChatMessage("system", self.system_prompt), ChatMessage("user", user_prompt, (image_path,))], generation, task="tx_free", hints=hints)
        return TxOutput(self.mode, result.text.strip(), [result.text], result.latency_seconds, None, [], user_prompt)


def build_tx(mode: str, backend: LLMBackend, cfg: dict[str, Any]) -> TxAgent:
    """Create the Tx agent for one mode from the ``tx`` config section."""
    if mode == "llm_free":
        return LLMFreeTx(backend, inform_budget=bool(cfg.get("inform_budget", True)))
    return TemplateTx(mode, backend, extractor=cfg.get("extractor", "backend"), max_retries=int(cfg.get("max_retries", 2)))
