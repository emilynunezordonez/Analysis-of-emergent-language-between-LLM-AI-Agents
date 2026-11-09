"""Deterministic ordinary-English representation."""

from transmitter.dataset.schema import ImageAttributes


def encode(attributes: ImageAttributes) -> str:
    count = len(attributes.objects)
    noun = "object" if count == 1 else "objects"
    descriptions = [f"{obj.size} {obj.colour} {obj.shape} located in the {obj.position}" for obj in attributes.objects]
    return f"There are {count} {noun}: " + "; ".join(descriptions) + "."
