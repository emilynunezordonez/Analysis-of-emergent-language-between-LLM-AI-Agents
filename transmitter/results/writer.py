"""CSV and JSON result persistence."""

import csv
import json
from pathlib import Path
from typing import Any


def write_records(records: list[dict[str, Any]], output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "messages.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    if records:
        with (root / "messages.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def write_analysis(analysis: dict[str, Any], output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
