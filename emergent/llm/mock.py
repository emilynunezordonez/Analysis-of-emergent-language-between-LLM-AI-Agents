"""Deterministic mock LLM for dry runs and tests (no GPU, no network).

IMPORTANT: everything this backend produces is *scripted*. Its messages, its
"convention drift" under memory, and its parsing errors are simulations built to
exercise the pipeline and analysis code. Results from mock runs are not
experimental findings about LLM communication.

Behaviour per task:

* ``tx_extract`` / ``tx_free``: the mock cannot see pixels, so it reads the
  dataset's ``metadata.json`` next to the image (an oracle, exactly like the
  existing ``transmitter.model.adapter.MockExtractor``).
* ``tx_free`` picks a verbosity level: 0 English, 1 telegraphic, 2 spaced codes,
  3 compact codes. It picks the most verbose level that fits an announced token
  budget, and drifts towards compact codes as shared history grows (simulated
  convention formation). ``temperature > 0`` adds lexical variation to English.
* ``rx``: a rule-based decoder that understands all mock levels and the three
  template formats (natural language, JSON, vector), including truncated ones.
  Missing attributes are filled with fixed guesses. With ``malformed_rate`` it
  sometimes emits invalid output, to exercise the receiver's retry logic.
* ``judge``: scores attribute agreement on a 0-10 scale.
"""

from __future__ import annotations

import json
import random
import re
import time
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import ChatMessage, Generation, GenerationConfig

SIZE_CODE = {"small": "S", "medium": "M", "large": "L"}
COLOUR_CODE = {"red": "R", "blue": "B", "green": "G", "yellow": "Y"}
SHAPE_CODE = {"circle": "c", "square": "s", "triangle": "t"}
POSITION_CODE = {"centre": "C", "upper-left": "UL", "upper-right": "UR", "lower-left": "LL", "lower-right": "LR"}
DEFAULT_GUESS = {"shape": "circle", "colour": "red", "position": "centre", "size": "medium"}
_COMPACT = re.compile(r"^([SML])([RBGY])([cst])(C|UL|UR|LL|LR)$")


class MockTokenizer:
    """Crude BPE-like tokenizer: one optional leading space plus a lowercase or capitalised
    word, at most two capitals, one digit, or one other character.

    Like real BPE vocabularies (Qwen's included), it keeps common words whole
    but splits codes such as ``MRcLL`` and digit strings into several tokens.
    Ids are stable CRC32 hashes; ``decode(encode(text)) == text`` holds for any input.
    """

    name = "mock-regex-tokenizer-v2"
    vocab_size: int | None = None
    _pattern = re.compile(r"\s?(?:[A-Z]?[a-z]+|[A-Z]{1,2}|\d|\S)|\s+")

    def __init__(self) -> None:
        self._id_to_token: dict[int, str] = {}

    def _pieces(self, text: str) -> list[str]:
        return self._pattern.findall(text)

    def encode(self, text: str) -> list[int]:
        ids = []
        for piece in self._pieces(text):
            token_id = zlib.crc32(piece.encode("utf-8")) & 0x7FFFFFFF
            self._id_to_token[token_id] = piece
            ids.append(token_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self._id_to_token[i] for i in ids)

    def token_strings(self, ids: Sequence[int]) -> list[str]:
        return [self._id_to_token[i] for i in ids]


@lru_cache(maxsize=None)
def _metadata(root: str) -> dict[str, list[dict[str, str]]]:
    records = json.loads((Path(root) / "metadata.json").read_text(encoding="utf-8"))
    return {record["image_id"]: record["objects"] for record in records}


def oracle_objects(image_path: Path) -> list[dict[str, str]]:
    """Ground-truth objects for a dataset image (``<root>/images/<id>.png``)."""
    return _metadata(str(image_path.parent.parent))[image_path.stem]


def _position_phrase(position: str, rng: random.Random | None) -> str:
    if position == "centre":
        return rng.choice(["in the centre", "at the centre", "in the middle"]) if rng else "in the centre"
    vertical, horizontal = position.split("-")
    if rng and rng.random() < 0.3:
        vertical = {"upper": "top", "lower": "bottom"}[vertical]
    return f"in the {vertical}-{horizontal} corner"


def render_level(objects: list[dict[str, str]], level: int, rng: random.Random | None = None) -> str:
    """Render objects at a mock verbosity level (0 English ... 3 compact codes)."""
    n = len(objects)
    if level == 0:
        plural = "s" if n != 1 else ""
        intro_options = [f"There {'are' if n != 1 else 'is'} {n} shape{plural} on a white background.", f"I see {n} shape{plural}.", f"The image contains {n} shape{plural}."]
        intro = rng.choice(intro_options) if rng else intro_options[0]
        verbs = ["sits", "is", "appears"]
        sentences = [f"A {o['size']} {o['colour']} {o['shape']} {rng.choice(verbs) if rng else 'sits'} {_position_phrase(o['position'], rng)}." for o in objects]
        return " ".join([intro, *sentences])
    if level == 1:
        return f"{n}: " + "; ".join(f"{o['size']} {o['colour']} {o['shape']} {o['position']}" for o in objects)
    if level == 2:
        return "; ".join(f"{SIZE_CODE[o['size']]} {COLOUR_CODE[o['colour']]} {SHAPE_CODE[o['shape']]} {POSITION_CODE[o['position']]}" for o in objects)
    return " ".join(SIZE_CODE[o["size"]] + COLOUR_CODE[o["colour"]] + SHAPE_CODE[o["shape"]] + POSITION_CODE[o["position"]] for o in objects)


_WORDS = {
    **{word: ("shape", word) for word in SHAPE_CODE},
    **{word: ("colour", word) for word in COLOUR_CODE},
    **{word: ("size", word) for word in SIZE_CODE},
    "big": ("size", "large"), "tiny": ("size", "small"),
}
_CODES = {
    **{code: ("size", word) for word, code in SIZE_CODE.items()},
    **{code: ("colour", word) for word, code in COLOUR_CODE.items()},
    **{code: ("shape", word) for word, code in SHAPE_CODE.items()},
    **{code: ("position", word) for word, code in POSITION_CODE.items()},
}
_POSITION_PATTERNS = [
    (re.compile(r"\b(?:upper|top)[\s-]+left\b", re.I), " upper-left "),
    (re.compile(r"\b(?:upper|top)[\s-]+right\b", re.I), " upper-right "),
    (re.compile(r"\b(?:lower|bottom)[\s-]+left\b", re.I), " lower-left "),
    (re.compile(r"\b(?:lower|bottom)[\s-]+right\b", re.I), " lower-right "),
    (re.compile(r"\b(?:centre|center|middle)\b", re.I), " centre "),
]


def _complete(partial: dict[str, str]) -> dict[str, str]:
    return {key: partial.get(key, DEFAULT_GUESS[key]) for key in ("shape", "colour", "position", "size")}


def mock_decode(message: str, max_objects: int = 10) -> list[dict[str, str]]:
    """Rule-based reconstruction of objects from any mock or template message."""
    text = message.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("objects"), list):
            return [_complete({k: v for k, v in o.items() if isinstance(v, str)}) for o in data["objects"] if isinstance(o, dict)][:max_objects]
        if isinstance(data, list) and all(isinstance(v, int) for v in data):
            return _decode_vector(data, max_objects)
    except json.JSONDecodeError:
        pass
    if re.fullmatch(r"\[\s*(\d+\s*,\s*)*\d*\s*\]?", text):
        return _decode_vector([int(v) for v in re.findall(r"\d+", text)], max_objects)
    for pattern, replacement in _POSITION_PATTERNS:
        text = pattern.sub(replacement, text)
    objects: list[dict[str, str]] = []
    for segment in re.split(r"[.;:|{}\n]", text):
        partial: dict[str, str] = {}
        for token in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", segment):
            compact = _COMPACT.match(token)
            if compact:
                size, colour, shape, position = (_CODES[g][1] for g in compact.groups())
                objects.append({"shape": shape, "colour": colour, "position": position, "size": size})
                continue
            if token.lower() in POSITION_CODE:
                partial["position"] = token.lower()
            elif token.lower() in _WORDS:
                key, value = _WORDS[token.lower()]
                partial[key] = value
            elif token in _CODES:
                key, value = _CODES[token]
                partial[key] = value
        if partial:
            objects.append(_complete(partial))
    return objects[:max_objects]


def _decode_vector(values: list[int], max_objects: int) -> list[dict[str, str]]:
    if not values:
        return []
    reverse = [{v: k for k, v in mapping.items()} for mapping in ({"circle": 1, "square": 2, "triangle": 3}, {"red": 1, "blue": 2, "green": 3, "yellow": 4}, {"centre": 1, "upper-left": 2, "upper-right": 3, "lower-left": 4, "lower-right": 5}, {"small": 1, "medium": 2, "large": 3})]
    count = min(values[0], max_objects)
    objects = []
    for index in range(count):
        block = values[1 + 4 * index: 5 + 4 * index]
        partial = {key: reverse[i][code] for i, (key, code) in enumerate(zip(("shape", "colour", "position", "size"), block)) if code in reverse[i]}
        objects.append(_complete(partial))
    return objects


class MockBackend:
    """Scripted stand-in for the shared LLM; see module docstring."""

    def __init__(self, malformed_rate: float = 0.0, model_id: str = "mock-llm-v1") -> None:
        self.model_id = model_id
        self.malformed_rate = malformed_rate
        self.tokenizer = MockTokenizer()

    def _level(self, objects: list[dict[str, str]], hints: Mapping[str, Any]) -> int:
        budget = hints.get("token_budget")
        level = 0
        if budget is not None:
            level = next((lv for lv in range(4) if len(self.tokenizer.encode(render_level(objects, lv))) <= budget), 3)
        history = int(hints.get("history_len", 0))
        return max(level, min(3, history // 4))

    def generate(
        self,
        messages: Sequence[ChatMessage],
        config: GenerationConfig,
        *,
        task: str,
        hints: Mapping[str, Any] | None = None,
    ) -> Generation:
        start = time.perf_counter()
        hints = dict(hints or {})
        rng = random.Random(config.seed if config.seed is not None else 0)
        images = [path for message in messages for path in message.images]
        if task == "tx_extract":
            text = json.dumps({"objects": oracle_objects(images[0])})
        elif task == "tx_free":
            objects = oracle_objects(images[0])
            text = render_level(objects, self._level(objects, hints), rng if config.temperature > 0 else None)
        elif task == "rx":
            text = self._rx(hints.get("message", ""), rng)
        elif task == "judge":
            text = self._judge(hints.get("source_objects", []), hints.get("reconstruction_objects", []))
        else:
            raise ValueError(f"unknown mock task {task!r}")
        return Generation(text, time.perf_counter() - start, len(self.tokenizer.encode(text)))

    def _rx(self, message: str, rng: random.Random) -> str:
        payload = json.dumps({"objects": mock_decode(message)})
        roll = rng.random()
        if roll < self.malformed_rate:
            return rng.choice([
                "I am not sure what the shapes are.",
                payload[: max(1, len(payload) // 2)],
                re.sub(r'"shape": "\w+"', '"shape": "hexagon"', payload, count=1) if '"shape"' in payload else "{}}",
            ])
        if roll < self.malformed_rate + 0.2:
            return f"Here is my reconstruction:\n```json\n{payload}\n```"
        return payload

    @staticmethod
    def _judge(source: list[dict[str, str]], reconstruction: list[dict[str, str]]) -> str:
        keys = ("shape", "colour", "position", "size")
        matched = sum(max((sum(s[k] == r[k] for k in keys) for r in reconstruction), default=0) for s in source)
        score = 10 * matched / (4 * max(len(source), len(reconstruction), 1))
        return json.dumps({"score": round(score, 1), "reason": "mock attribute agreement"})
