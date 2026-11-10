"""Robust parsing and validation of Rx output into the shared attribute schema.

The schema is the same one the Tx dataset uses (``transmitter.dataset.schema``),
so a reconstruction renders with exactly the same drawing code as the source.
Common harmless variants (code fences, prose around the JSON, ``color``,
``center``, ``top-left``...) are normalised. Anything else is reported as an
error so the agent can retry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from transmitter.dataset.schema import COLOURS, POSITIONS, SHAPES, SIZES, ShapeObject

ALLOWED: dict[str, tuple[str, ...]] = {"shape": SHAPES, "colour": COLOURS, "position": POSITIONS, "size": SIZES}
_KEY_SYNONYMS = {"color": "colour", "type": "shape", "location": "position", "pos": "position"}
_VALUE_SYNONYMS = {
    "center": "centre", "middle": "centre", "centre": "centre",
    "top-left": "upper-left", "top-right": "upper-right", "bottom-left": "lower-left", "bottom-right": "lower-right",
    "left-upper": "upper-left", "right-upper": "upper-right", "left-lower": "lower-left", "right-lower": "lower-right",
    "big": "large",
}


@dataclass
class ParseResult:
    """Valid objects found, plus errors. ``valid`` is True only if there were no errors."""

    objects: list[ShapeObject] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def objects_as_dicts(self) -> list[dict[str, str]]:
        return [obj.to_dict() for obj in self.objects]


def extract_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` or ``[...]`` block in ``text``, ignoring braces inside strings."""
    start = next((i for i, ch in enumerate(text) if ch in "{["), None)
    if start is None:
        return None
    stack: list[str] = []
    in_string = escaped = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                return None
            if not stack:
                return text[start: index + 1]
    return None


def _normalise_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    cleaned = "-".join(value.strip().lower().replace("_", " ").split())
    return _VALUE_SYNONYMS.get(cleaned, cleaned)


def parse_reconstruction(text: str, max_objects: int = 10) -> ParseResult:
    """Parse raw model output. Invalid objects are skipped and reported in ``errors``."""
    result = ParseResult()
    block = extract_json_object(text)
    if block is None:
        result.errors.append("no complete JSON object found")
        return result
    try:
        data = json.loads(block)
    except json.JSONDecodeError as error:
        result.errors.append(f"invalid JSON: {error.msg}")
        return result
    items = data.get("objects") if isinstance(data, dict) else data
    if not isinstance(items, list):
        result.errors.append('expected {"objects": [...]}')
        return result
    if len(items) > max_objects:
        result.errors.append(f"too many objects ({len(items)} > {max_objects}); extra objects dropped")
        items = items[:max_objects]
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            result.errors.append(f"object {index} is not a JSON object")
            continue
        normalised = {_KEY_SYNONYMS.get(str(k).strip().lower(), str(k).strip().lower()): _normalise_value(v) for k, v in item.items()}
        problems = []
        for key, allowed in ALLOWED.items():
            if key not in normalised:
                problems.append(f"object {index} is missing '{key}'")
            elif normalised[key] not in allowed:
                problems.append(f"object {index} has invalid {key}={normalised[key]!r} (allowed: {', '.join(allowed)})")
        if problems:
            result.errors.extend(problems)
            continue
        result.objects.append(ShapeObject(**{key: normalised[key] for key in ALLOWED}))
    return result
