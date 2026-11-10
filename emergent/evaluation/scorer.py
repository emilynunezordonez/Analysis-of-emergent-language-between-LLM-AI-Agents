"""Independent evaluators: Tx and Rx never grade themselves.

``ProgrammaticScorer`` is the primary evaluator. ``LLMJudge`` is an optional
third agent, enabled with ``evaluation.llm_judge: true``. It is shown both
images (it is a vision model) and returns a 0-10 score. If it shares weights
with Tx/Rx, the scores may be biased towards the model's own conventions; set
``evaluation.judge_model_id`` to load a different model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from emergent.llm.base import ChatMessage, GenerationConfig, LLMBackend
from emergent.rx.parsing import extract_json_object

from .metrics import attribute_metrics, colour_metrics, mse, ssim

_JUDGE_PROMPT = (
    "You are an impartial judge. The first image is the original; the second is a reconstruction "
    "made by an agent that only read a text description. Rate how well the reconstruction matches the "
    'original on a 0-10 scale (shape types, colours, sizes, positions, number of shapes). Reply ONLY with JSON: {"score": <number>, "reason": "<short>"}'
)


class ProgrammaticScorer:
    """Pixel, colour and symbolic metrics. Deterministic and free."""

    name = "programmatic-v1"

    def score(self, source: Image.Image, reconstruction: Image.Image, source_objects: list[dict[str, str]], reconstructed_objects: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "ssim": ssim(source, reconstruction),
            "mse": mse(source, reconstruction),
            **colour_metrics(source, reconstruction),
            **attribute_metrics(source_objects, reconstructed_objects),
        }


class LLMJudge:
    """Optional LLM-as-judge (third agent). Failures produce ``None``, never a crash."""

    def __init__(self, backend: LLMBackend, generation: GenerationConfig | None = None) -> None:
        self.backend = backend
        self.generation = generation or GenerationConfig(temperature=0.0, max_new_tokens=128)
        self.name = f"llm-judge:{backend.model_id}"

    def score(self, source_path: Path, reconstruction_path: Path, source_objects: list[dict[str, str]], reconstructed_objects: list[dict[str, str]]) -> dict[str, Any]:
        hints = {"source_objects": source_objects, "reconstruction_objects": reconstructed_objects}  # mock only
        try:
            result = self.backend.generate([ChatMessage("user", _JUDGE_PROMPT, (source_path, reconstruction_path))], self.generation, task="judge", hints=hints)
            data = json.loads(extract_json_object(result.text) or "")
            return {"judge_score": float(data["score"]), "judge_reason": str(data.get("reason", "")), "judge_raw": result.text}
        except Exception as error:  # noqa: BLE001 - the judge is advisory
            return {"judge_score": None, "judge_reason": f"judge failed: {type(error).__name__}: {error}", "judge_raw": None}
