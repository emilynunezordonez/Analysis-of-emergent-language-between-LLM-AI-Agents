"""YAML config with defaults and ``key.path=value`` overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "experiment": {"name": "experiment", "output_dir": "runs/experiment", "seeds": [0], "rounds": 1, "shuffle": True, "max_episodes": None},
    "dataset": {"data_dir": "data", "count": 20, "seed": 7, "limit": None},
    "backend": {"type": "mock", "model_id": "Qwen/Qwen2.5-VL-3B-Instruct", "device_map": "auto", "torch_dtype": "auto", "mock": {"malformed_rate": 0.0}},
    "tx": {"modes": ["llm_free"], "extractor": "backend", "inform_budget": True, "max_retries": 2, "generation": {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 256}},
    "channel": {"token_budgets": [None], "noise": {"name": "identity", "params": {}}},
    "rx": {"max_retries": 2, "max_objects": 10, "generation": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 256}},
    "memory": {"enabled": [False], "window": 5, "tx_feedback": "score", "rx_feedback": "ground_truth"},
    "evaluation": {"llm_judge": False, "judge_model_id": None},
    "logging": {"save_images": 3},
    "analysis": {"reference_corpus": "data/reference/carroll-alice.txt", "bootstrap": 1000, "baseline_messages": 200},
}


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        merged[key] = deep_merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else copy.deepcopy(value)
    return merged


def set_path(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    node = cfg
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b=value`` strings; values are parsed as YAML (``[null, 10]``, ``true``, ``0.7``)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        set_path(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        cfg = deep_merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    cfg = apply_overrides(cfg, overrides or [])
    for dotted in ("experiment.seeds", "tx.modes", "channel.token_budgets", "memory.enabled"):
        section, key = dotted.split(".")
        cfg[section][key] = as_list(cfg[section][key])
    return cfg
