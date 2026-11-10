"""Render a reconstruction with exactly the drawing code used for the source images."""

from __future__ import annotations

from typing import Iterable

from PIL import Image, ImageDraw

from transmitter.dataset.generator import _draw_object  # same renderer as the dataset, so pixels are comparable
from transmitter.dataset.schema import ShapeObject

CANVAS = (256, 256)  # the coordinate system of the dataset generator


def render(objects: Iterable[ShapeObject], size: tuple[int, int] = CANVAS) -> Image.Image:
    """Draw objects in order on a white canvas; resize if the source resolution differs."""
    image = Image.new("RGB", CANVAS, "white")
    draw = ImageDraw.Draw(image)
    for obj in objects:
        _draw_object(draw, obj)
    return image if size == CANVAS else image.resize(size, Image.NEAREST)
