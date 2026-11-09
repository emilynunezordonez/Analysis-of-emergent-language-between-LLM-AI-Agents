"""Dataset loading and integrity checks."""

import json
from pathlib import Path

from .schema import ImageAttributes


def load_dataset(root: str | Path) -> list[tuple[Path, ImageAttributes]]:
    root_path = Path(root)
    raw_records = json.loads((root_path / "metadata.json").read_text(encoding="utf-8"))
    records: list[tuple[Path, ImageAttributes]] = []
    seen: set[str] = set()
    for raw in raw_records:
        attributes = ImageAttributes.from_dict(raw)
        if attributes.image_id in seen:
            raise ValueError(f"duplicate image ID: {attributes.image_id}")
        seen.add(attributes.image_id)
        image_path = root_path / "images" / f"{attributes.image_id}.png"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        records.append((image_path, attributes))
    if not records:
        raise ValueError("dataset is empty")
    return records
