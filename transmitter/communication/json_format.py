"""Canonical JSON representation."""

import json

from transmitter.dataset.schema import ImageAttributes


def encode(attributes: ImageAttributes) -> str:
    return json.dumps({"objects": [obj.to_dict() for obj in attributes.objects]}, separators=(",", ":"), ensure_ascii=True)


def decode(message: str) -> list[dict[str, str]]:
    data = json.loads(message)
    if not isinstance(data, dict) or not isinstance(data.get("objects"), list):
        raise ValueError("JSON message must contain an objects list")
    return data["objects"]
