"""Communication loop: for each condition x seed x round x image, Tx -> channel -> Rx -> render -> score -> log.

A *condition* is (tx_mode, token_budget, memory on/off). A *game* is one
condition with one seed; memory is reset at the start of every game. An
*episode* is one image. Each episode gets a stable seed derived from
(seed, condition, round, image), so any episode can be reproduced alone.

Outputs in ``experiment.output_dir``:

* ``manifest.json``: resolved config, model parameters, git commit, package versions
* ``episodes.jsonl``: one JSON object per episode (see README, "Episode log schema")
* ``failures.jsonl``: episodes with Tx extraction errors or invalid Rx output
* ``reconstructions/``: rendered reconstructions (``logging.save_images`` per game)

Usage::

    python -m emergent.runner.run --config configs/dry_run.yaml
    python -m emergent.runner.run --config configs/qwen_budget_ladder.yaml --seeds 0 1 2 --output-dir runs/ladder
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import random
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from transmitter.dataset.generator import generate_dataset
from transmitter.dataset.loader import load_dataset
from transmitter.dataset.schema import ImageAttributes, ShapeObject

from emergent.channel import Channel
from emergent.evaluation import LLMJudge, ProgrammaticScorer
from emergent.llm import GenerationConfig, LLMBackend, backend_info, build_backend
from emergent.memory import EpisodeRecap, SharedMemory
from emergent.rx import RxAgent, render
from emergent.tx import TX_MODES, build_tx

from .config import load_config


def stable_seed(*parts: Any) -> int:
    """Seed derived from parts; stable across processes, unlike ``hash``."""
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8")) & 0x7FFFFFFF


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def condition_id(mode: str, budget: int | None, memory: bool) -> str:
    return f"{mode}__budget-{'inf' if budget is None else budget}__mem-{'on' if memory else 'off'}"


def source_label(attributes: ImageAttributes) -> str:
    return " | ".join(f"{o.size} {o.colour} {o.shape} @ {o.position}" for o in attributes.objects)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "pillow", "scikit-image", "scipy", "pyyaml", "matplotlib", "torch", "transformers", "qwen-vl-utils", "accelerate"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return versions


def _generation(section: dict[str, Any], seed: int) -> GenerationConfig:
    return GenerationConfig(float(section.get("temperature", 0.0)), float(section.get("top_p", 1.0)), int(section.get("max_new_tokens", 256)), seed)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def run_experiment(cfg: dict[str, Any], backend: LLMBackend | None = None) -> Path:
    """Run every configured condition and return the output directory."""
    exp, data_cfg = cfg["experiment"], cfg["dataset"]
    out_dir = Path(exp["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_cfg["data_dir"])
    if not (data_dir / "metadata.json").exists():
        generate_dataset(data_dir, count=int(data_cfg["count"]), seed=int(data_cfg["seed"]))
    dataset = load_dataset(data_dir)
    if data_cfg.get("limit"):
        dataset = dataset[: int(data_cfg["limit"])]
    for mode in cfg["tx"]["modes"]:
        if mode not in TX_MODES:
            raise ValueError(f"unknown tx mode {mode!r}; choose from {TX_MODES}")

    backend = backend or build_backend(cfg["backend"])
    info = backend_info(cfg["backend"], backend)
    judge = None
    if cfg["evaluation"].get("llm_judge"):
        judge_model = cfg["evaluation"].get("judge_model_id")
        judge_backend = build_backend({**cfg["backend"], "model_id": judge_model}) if judge_model else backend
        judge = LLMJudge(judge_backend)
    scorer = ProgrammaticScorer()
    run_id = f"{exp['name']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    manifest = {
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "model": asdict(info),
        "evaluator": scorer.name,
        "judge": judge.name if judge else None,
        "dataset": {"data_dir": _relative(data_dir), "n_images": len(dataset)},
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _versions(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    save_images = int(cfg["logging"].get("save_images", 3))
    tmp_dir = Path(tempfile.mkdtemp(prefix="rx-judge-"))
    conditions = list(itertools.product(cfg["tx"]["modes"], cfg["channel"]["token_budgets"], cfg["memory"]["enabled"]))
    n_written = 0
    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as episodes_file, (out_dir / "failures.jsonl").open("w", encoding="utf-8") as failures_file:
        for mode, budget, memory_on in conditions:
            cond = condition_id(mode, budget, bool(memory_on))
            tx = build_tx(mode, backend, cfg["tx"])
            rx = RxAgent(backend, int(cfg["rx"]["max_retries"]), int(cfg["rx"]["max_objects"]))
            channel = Channel(backend.tokenizer, budget, cfg["channel"]["noise"]["name"], cfg["channel"]["noise"].get("params"))
            for seed in exp["seeds"]:
                seed_everything(int(seed))
                memory = SharedMemory(bool(memory_on), int(cfg["memory"]["window"]), cfg["memory"]["tx_feedback"], cfg["memory"]["rx_feedback"])
                order_rng = random.Random(stable_seed(seed, cond, "order"))
                episode_index = 0
                for round_index in range(int(exp["rounds"])):
                    images = list(dataset)
                    if exp.get("shuffle", True):
                        order_rng.shuffle(images)
                    for image_path, ground_truth in images:
                        if exp.get("max_episodes") is not None and episode_index >= int(exp["max_episodes"]):
                            break
                        record = run_episode(
                            run_id=run_id, cond=cond, mode=mode, budget=budget, seed=int(seed), round_index=round_index, episode_index=episode_index,
                            image_path=image_path, ground_truth=ground_truth, tx=tx, rx=rx, channel=channel, memory=memory, scorer=scorer, judge=judge,
                            cfg=cfg, info=asdict(info), image_dir=out_dir / "reconstructions" / cond / f"seed{seed}" if episode_index < save_images or save_images < 0 else None, tmp_dir=tmp_dir,
                        )
                        episodes_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                        if record["tx"]["errors"] or not record["rx"]["valid"]:
                            failures_file.write(json.dumps({"episode_id": record["episode_id"], "tx_errors": record["tx"]["errors"], "rx_errors": record["rx"]["errors"], "rx_fallback": record["rx"]["fallback"], "rx_raw_outputs": record["rx"]["raw_outputs"]}, ensure_ascii=False) + "\n")
                        episode_index += 1
                        n_written += 1
                episodes_file.flush()
            print(f"[{cond}] done ({n_written} episodes so far)", flush=True)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["n_episodes"] = n_written
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return out_dir


def run_episode(*, run_id: str, cond: str, mode: str, budget: int | None, seed: int, round_index: int, episode_index: int, image_path: Path, ground_truth: ImageAttributes,
                tx: Any, rx: RxAgent, channel: Channel, memory: SharedMemory, scorer: ProgrammaticScorer, judge: LLMJudge | None,
                cfg: dict[str, Any], info: dict[str, Any], image_dir: Path | None, tmp_dir: Path) -> dict[str, Any]:
    """One Tx -> channel -> Rx exchange. Rx only receives ``channel_out.received``."""
    episode_seed = stable_seed(seed, cond, round_index, ground_truth.image_id)
    tx_gen, rx_gen = _generation(cfg["tx"]["generation"], episode_seed), _generation(cfg["rx"]["generation"], episode_seed + 1)
    memory_size = len(memory)
    start = time.perf_counter()
    tx_out = tx.transmit(image_path, ground_truth, token_budget=budget, memory=memory, generation=tx_gen)
    channel_out = channel.transmit(tx_out.message, random.Random(episode_seed + 2))
    rx_out = rx.receive(channel_out.received, memory=memory, generation=rx_gen)
    total_latency = time.perf_counter() - start

    source_image = Image.open(image_path).convert("RGB")
    reconstruction = render([ShapeObject(**o) for o in rx_out.objects], source_image.size)
    source_objects = [o.to_dict() for o in ground_truth.objects]
    metrics = scorer.score(source_image, reconstruction, source_objects, rx_out.objects)
    episode_id = f"{cond}/seed{seed}/ep{episode_index:04d}"
    reconstruction_path = None
    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)
        reconstruction_path = image_dir / f"ep{episode_index:04d}_{ground_truth.image_id}.png"
        reconstruction.save(reconstruction_path)
    judge_result = None
    if judge is not None:
        judge_path = reconstruction_path or tmp_dir / "reconstruction.png"
        if reconstruction_path is None:
            reconstruction.save(judge_path)
        judge_result = judge.score(image_path, judge_path, source_objects, rx_out.objects)
    memory.add(EpisodeRecap(episode_index, tx_out.message, channel_out.received, rx_out.objects, source_objects, metrics["attribute_score"]))

    tokenizer = channel.tokenizer
    return {
        "run_id": run_id,
        "episode_id": episode_id,
        "condition_id": cond,
        "condition": {"tx_mode": mode, "token_budget": budget, "memory": memory.enabled, "noise": channel.noise},
        "seed": seed,
        "episode_seed": episode_seed,
        "round": round_index,
        "episode_index": episode_index,
        "image_id": ground_truth.image_id,
        "image_path": _relative(image_path),
        "source_label": source_label(ground_truth),
        "source_objects": source_objects,
        "tx": {
            "mode": mode,
            "raw_outputs": tx_out.raw_outputs,
            "message": tx_out.message,
            "token_ids": channel_out.sent_token_ids,
            "tokens": tokenizer.token_strings(channel_out.sent_token_ids),
            "token_count": len(channel_out.sent_token_ids),
            "char_count": len(tx_out.message),
            "word_count": len(tx_out.message.split()),
            "extracted_objects": tx_out.extracted_objects,
            "errors": tx_out.errors,
            "prompt": tx_out.prompt,
            "latency_seconds": tx_out.latency_seconds,
        },
        "channel": {
            "token_budget": budget,
            "noise": channel.noise,
            "noise_params": channel.noise_params,
            "truncated": channel_out.truncated,
            "received_message": channel_out.received,
            "received_token_ids": channel_out.received_token_ids,
            "received_token_count": len(channel_out.received_token_ids),
            "received_char_count": len(channel_out.received),
        },
        "rx": {
            "raw_outputs": rx_out.raw_outputs,
            "attempts": rx_out.attempts,
            "valid": rx_out.valid,
            "fallback": rx_out.fallback,
            "errors": rx_out.errors,
            "reconstruction": rx_out.objects,
            "reconstruction_path": _relative(reconstruction_path) if reconstruction_path else None,
            "latency_seconds": rx_out.latency_seconds,
        },
        "metrics": metrics,
        "judge": judge_result,
        "latency": {"tx_seconds": tx_out.latency_seconds, "rx_seconds": rx_out.latency_seconds, "episode_wall_seconds": total_latency},
        "model": {**info, "tx_generation": tx_gen.as_dict(), "rx_generation": rx_gen.as_dict()},
        "memory": {**memory.config(), "size_at_start": memory_size},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Tx -> channel -> Rx reconstruction game.")
    parser.add_argument("--config", type=Path, help="YAML config (see configs/)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="override any config key, e.g. --set channel.token_budgets=[null,10]")
    parser.add_argument("--dry-run", action="store_true", help="force the mock LLM (no GPU/network)")
    parser.add_argument("--model", help="backend.model_id")
    parser.add_argument("--temperature", type=float, help="Tx sampling temperature")
    parser.add_argument("--rounds", type=int, help="passes over the dataset per game")
    parser.add_argument("--episodes", type=int, help="max episodes per game (condition x seed)")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--token-budgets", nargs="+", help="e.g. inf 50 20 10 5")
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides = list(args.overrides)
    if args.dry_run:
        overrides.append("backend.type=mock")
    if args.model:
        overrides.append(f"backend.model_id={args.model}")
    if args.temperature is not None:
        overrides.append(f"tx.generation.temperature={args.temperature}")
    if args.rounds is not None:
        overrides.append(f"experiment.rounds={args.rounds}")
    if args.episodes is not None:
        overrides.append(f"experiment.max_episodes={args.episodes}")
    if args.seeds:
        overrides.append(f"experiment.seeds={json.dumps(args.seeds)}")
    if args.token_budgets:
        budgets = [None if b.lower() in ("inf", "none", "null") else int(b) for b in args.token_budgets]
        overrides.append(f"channel.token_budgets={json.dumps(budgets)}")
    if args.output_dir:
        overrides.append(f"experiment.output_dir={args.output_dir}")
    return load_config(args.config, overrides)


def main(argv: list[str] | None = None) -> None:
    out_dir = run_experiment(config_from_args(parse_args(argv)))
    print(f"Episodes written to {out_dir / 'episodes.jsonl'}")


if __name__ == "__main__":
    main()
