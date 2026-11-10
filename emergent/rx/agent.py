"""Receiver agent: text message in, structured reconstruction out. Rx never sees an image.

Output format: JSON in the dataset's attribute schema, rendered by the dataset's
own renderer (``emergent.rx.render``). We chose this over free SVG because
(1) source and reconstruction then share one pixel renderer, so pixel metrics
measure communication errors rather than drawing-style differences; (2) shape,
colour, size and position accuracy can be scored exactly against the ground
truth; (3) validation and retries are simple and well defined. The cost is that
Rx cannot express anything outside the schema. For future non-synthetic sources
(photos), swap in an SVG/free-form decoder behind the same ``RxOutput``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from emergent.llm.base import ChatMessage, GenerationConfig, LLMBackend
from emergent.memory import SharedMemory

from .parsing import ParseResult, parse_reconstruction

_SYSTEM_PROMPT = Path(__file__).with_name("prompts") / "system.txt"


@dataclass
class RxOutput:
    """All attempts and the final reconstruction for one episode.

    ``fallback`` is ``None`` (first valid answer used), ``"partial"`` (no attempt
    was fully valid; the valid objects of the best attempt are kept) or
    ``"empty"`` (nothing usable; a blank canvas is rendered).
    """

    raw_outputs: list[str]
    objects: list[dict[str, str]]
    valid: bool
    attempts: int
    errors: list[str] = field(default_factory=list)
    fallback: str | None = None
    latency_seconds: float = 0.0
    prompt: str = ""


class RxAgent:
    """Asks the shared backend for a JSON reconstruction, validating and retrying."""

    def __init__(self, backend: LLMBackend, max_retries: int = 2, max_objects: int = 10, system_prompt_path: Path = _SYSTEM_PROMPT) -> None:
        self.backend = backend
        self.max_retries = max_retries
        self.max_objects = max_objects
        self.system_prompt = system_prompt_path.read_text(encoding="utf-8")

    @staticmethod
    def build_user_prompt(message: str, memory: SharedMemory) -> str:
        parts = []
        history = memory.rx_view()
        if history:
            parts.append("Recent episodes with the same partner:\n" + history)
        parts.append(f"Message from the transmitter:\n<<<\n{message}\n>>>\n\nReply with the JSON object only.")
        return "\n\n".join(parts)

    def receive(self, message: str, *, memory: SharedMemory, generation: GenerationConfig) -> RxOutput:
        """Never raises on bad model output; failures are recorded in the result."""
        user_prompt = self.build_user_prompt(message, memory)
        chat = [ChatMessage("system", self.system_prompt), ChatMessage("user", user_prompt)]
        raws: list[str] = []
        errors: list[str] = []
        parsed: list[ParseResult] = []
        latency = 0.0
        for attempt in range(self.max_retries + 1):
            seed = None if generation.seed is None else generation.seed + 1000 * attempt
            config = GenerationConfig(generation.temperature, generation.top_p, generation.max_new_tokens, seed)
            try:
                result = self.backend.generate(chat, config, task="rx", hints={"message": message})
            except Exception as error:  # a backend crash must not kill the run
                errors.append(f"attempt {attempt}: backend error {type(error).__name__}: {error}")
                continue
            latency += result.latency_seconds
            raws.append(result.text)
            outcome = parse_reconstruction(result.text, self.max_objects)
            parsed.append(outcome)
            if outcome.valid:
                return RxOutput(raws, outcome.objects_as_dicts(), True, attempt + 1, errors, None, latency, user_prompt)
            errors.append(f"attempt {attempt}: " + "; ".join(outcome.errors))
            chat = [*chat, ChatMessage("assistant", result.text), ChatMessage("user", "Your reply was invalid: " + "; ".join(outcome.errors) + ". Reply again with ONLY the JSON object in the required format.")]
        best = max(parsed, key=lambda p: len(p.objects), default=None)
        if best is not None and best.objects:
            return RxOutput(raws, best.objects_as_dicts(), False, len(raws), errors, "partial", latency, user_prompt)
        return RxOutput(raws, [], False, len(raws), errors, "empty", latency, user_prompt)


