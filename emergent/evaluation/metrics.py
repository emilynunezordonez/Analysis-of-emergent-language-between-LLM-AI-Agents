"""Programmatic metrics comparing a source image with its reconstruction.

Pixel level: SSIM (RGB, data range 255) and MSE on [0, 1] intensities.
Colour: dominant non-background colour of each image; CIE76 Delta E in Lab.
Symbolic level (the ground truth is known for synthetic shapes): objects are
matched one-to-one by maximum attribute agreement (Hungarian algorithm), then
shape/colour/size/position accuracy, count accuracy and exact match are computed.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity

ATTRIBUTES = ("shape", "colour", "size", "position")
PALETTE = {"red": (220, 45, 45), "blue": (45, 90, 210), "green": (45, 170, 80), "yellow": (230, 190, 35)}
_BACKGROUND_THRESHOLD = 245


def _array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def ssim(source: Image.Image, reconstruction: Image.Image) -> float:
    return float(structural_similarity(_array(source), _array(reconstruction), channel_axis=2, data_range=255))


def mse(source: Image.Image, reconstruction: Image.Image) -> float:
    a = _array(source).astype(np.float64) / 255.0
    b = _array(reconstruction).astype(np.float64) / 255.0
    return float(np.mean((a - b) ** 2))


def dominant_colour(image: Image.Image) -> tuple[int, int, int] | None:
    """Most frequent non-white pixel colour, or ``None`` for a blank canvas."""
    pixels = _array(image).reshape(-1, 3)
    foreground = pixels[np.any(pixels < _BACKGROUND_THRESHOLD, axis=1)]
    if len(foreground) == 0:
        return None
    colours, counts = np.unique(foreground, axis=0, return_counts=True)
    return tuple(int(v) for v in colours[int(np.argmax(counts))])


def delta_e(rgb_a: Sequence[int], rgb_b: Sequence[int]) -> float:
    lab = rgb2lab(np.array([[rgb_a, rgb_b]], dtype=np.float64) / 255.0)[0]
    return float(np.linalg.norm(lab[0] - lab[1]))


def nearest_colour_name(rgb: Sequence[int]) -> str:
    return min(PALETTE, key=lambda name: delta_e(rgb, PALETTE[name]))


def colour_metrics(source: Image.Image, reconstruction: Image.Image) -> dict[str, Any]:
    a, b = dominant_colour(source), dominant_colour(reconstruction)
    if a is None or b is None:
        return {"dominant_colour_source": a, "dominant_colour_reconstruction": b, "dominant_colour_delta_e": None, "dominant_colour_match": a is None and b is None}
    return {
        "dominant_colour_source": list(a),
        "dominant_colour_reconstruction": list(b),
        "dominant_colour_delta_e": delta_e(a, b),
        "dominant_colour_match": nearest_colour_name(a) == nearest_colour_name(b),
    }


def match_objects(source: list[dict[str, str]], reconstruction: list[dict[str, str]]) -> list[tuple[int, int]]:
    """One-to-one matching maximising the number of agreeing attributes."""
    if not source or not reconstruction:
        return []
    agreement = np.array([[sum(s[k] == r[k] for k in ATTRIBUTES) for r in reconstruction] for s in source])
    rows, cols = linear_sum_assignment(-agreement)
    return list(zip(rows.tolist(), cols.tolist()))


def attribute_metrics(source: list[dict[str, str]], reconstruction: list[dict[str, str]]) -> dict[str, Any]:
    """Per-attribute accuracy over source objects; unmatched source objects count as wrong."""
    pairs = match_objects(source, reconstruction)
    n_source, n_rec = len(source), len(reconstruction)
    metrics: dict[str, Any] = {"n_source_objects": n_source, "n_reconstructed_objects": n_rec, "count_correct": n_source == n_rec}
    for key in ATTRIBUTES:
        metrics[f"{key}_accuracy"] = sum(source[i][key] == reconstruction[j][key] for i, j in pairs) / n_source if n_source else float(n_rec == 0)
    agreements = sum(sum(source[i][k] == reconstruction[j][k] for k in ATTRIBUTES) for i, j in pairs)
    denominator = len(ATTRIBUTES) * max(n_source, n_rec)
    metrics["attribute_score"] = agreements / denominator if denominator else 1.0
    exact_pairs = sum(all(source[i][k] == reconstruction[j][k] for k in ATTRIBUTES) for i, j in pairs)
    metrics["object_recall"] = exact_pairs / n_source if n_source else float(n_rec == 0)
    metrics["object_precision"] = exact_pairs / n_rec if n_rec else float(n_source == 0)
    as_multiset = lambda objects: Counter(tuple(o[k] for k in ATTRIBUTES) for o in objects)
    metrics["exact_match"] = as_multiset(source) == as_multiset(reconstruction)
    return metrics
