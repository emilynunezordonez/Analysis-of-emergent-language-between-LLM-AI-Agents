"""Deterministic geometric PNG dataset generator."""

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from .schema import COLOURS, POSITIONS, SHAPES, SIZES, ImageAttributes, ShapeObject

_COLOUR_RGB = {"red": (220, 45, 45), "blue": (45, 90, 210), "green": (45, 170, 80), "yellow": (230, 190, 35)}
_POSITION_CENTRES = {"centre": (128, 128), "upper-left": (64, 64), "upper-right": (192, 64), "lower-left": (64, 192), "lower-right": (192, 192)}
_SIZE_RADIUS = {"small": 22, "medium": 38, "large": 55}


def _draw_object(draw: ImageDraw.ImageDraw, obj: ShapeObject) -> None:
    x, y = _POSITION_CENTRES[obj.position]
    radius = _SIZE_RADIUS[obj.size]
    colour = _COLOUR_RGB[obj.colour]
    bounds = (x - radius, y - radius, x + radius, y + radius)
    if obj.shape == "circle":
        draw.ellipse(bounds, fill=colour)
    elif obj.shape == "square":
        draw.rectangle(bounds, fill=colour)
    else:
        draw.polygon([(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)], fill=colour)


def generate_dataset(output_dir: str | Path, count: int = 20, seed: int = 7) -> list[ImageAttributes]:
    """Generate PNGs and metadata with stable IDs and deterministic object order."""
    if not 1 <= count <= 50:
        raise ValueError("count must be between 1 and 50")
    root = Path(output_dir)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records: list[ImageAttributes] = []
    for index in range(count):
        object_count = 1 + (index % 3)
        objects = tuple(
            ShapeObject(
                shape=rng.choice(SHAPES),
                colour=rng.choice(COLOURS),
                position=rng.choice(POSITIONS),
                size=rng.choice(SIZES),
            )
            for _ in range(object_count)
        )
        attributes = ImageAttributes(f"image_{index:03d}", objects)
        image = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(image)
        for obj in attributes.objects:
            _draw_object(draw, obj)
        image.save(image_dir / f"{attributes.image_id}.png")
        records.append(attributes)
    (root / "metadata.json").write_text(json.dumps([item.to_dict() for item in records], indent=2), encoding="utf-8")
    (root / "generation_config.json").write_text(json.dumps({"count": count, "seed": seed}, indent=2), encoding="utf-8")
    return records
