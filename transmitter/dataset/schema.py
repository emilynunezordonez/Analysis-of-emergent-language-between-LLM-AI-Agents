"""Validated image-attribute schema shared by every communication mode."""

from dataclasses import asdict, dataclass
from typing import Any

SHAPES = ("circle", "square", "triangle")
COLOURS = ("red", "blue", "green", "yellow")
POSITIONS = ("centre", "upper-left", "upper-right", "lower-left", "lower-right")
SIZES = ("small", "medium", "large")


@dataclass(frozen=True)
class ShapeObject:
    shape: str
    colour: str
    position: str
    size: str

    def __post_init__(self) -> None:
        for field_name, allowed in (("shape", SHAPES), ("colour", COLOURS), ("position", POSITIONS), ("size", SIZES)):
            value = getattr(self, field_name)
            if value not in allowed:
                raise ValueError(f"Invalid {field_name}={value!r}; expected one of {allowed}")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ImageAttributes:
    image_id: str
    objects: tuple[ShapeObject, ...]

    def __post_init__(self) -> None:
        if not self.image_id.strip():
            raise ValueError("image_id must not be empty")
        if not self.objects:
            raise ValueError("an image must contain at least one object")

    def to_dict(self) -> dict[str, Any]:
        return {"image_id": self.image_id, "objects": [obj.to_dict() for obj in self.objects]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageAttributes":
        return cls(
            image_id=str(data["image_id"]),
            objects=tuple(ShapeObject(**item) for item in data["objects"]),
        )
