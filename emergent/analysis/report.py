"""Per-condition language measures, task performance, English baselines, and figures.

Usage::

    python -m emergent.analysis.report --run-dir runs/sample
    python -m emergent.analysis.report --run-dir runs/a runs/b --out-dir reports/combined

Writes ``tables/summary.csv`` (+ ``summary.md``), ``tables/rounds.csv`` (when a
run has more than one round) and ``figures/*.png|pdf`` in ``--out-dir``
(default: ``<first run dir>/analysis``).

Analysed text: by default the *received* message (what actually crossed the
channel, after budget truncation); use ``--field sent`` for Tx's full output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from emergent.llm import build_tokenizer

from .baselines import reference_corpus, template_english
from .measures import describe

PERFORMANCE_KEYS = ("attribute_score", "ssim", "mse", "exact_match", "shape_accuracy", "colour_accuracy", "size_accuracy", "position_accuracy", "count_correct", "dominant_colour_match")
CAVEAT = (
    "Entropy estimates from small samples are biased (plug-in estimates are too low). Values use the Miller-Madow correction "
    "with 95% basic-bootstrap CIs over messages. Conditional (bigram) entropies need far more data than unigrams: check "
    "`token_bigram_samples_per_type`. Rows flagged `undersampled` have fewer than 10 tokens per token type and should not be "
    "interpreted. Unigram entropy per token is not information per message: also compare `est_bits_per_message_unigram` and "
    "compression bytes per message."
)


def load_episodes(run_dirs: Iterable[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {}
    for run_dir in run_dirs:
        manifest = manifest or json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        with (run_dir / "episodes.jsonl").open(encoding="utf-8") as handle:
            episodes.extend(json.loads(line) for line in handle if line.strip())
    return episodes, manifest


def _message(episode: dict[str, Any], field: str) -> tuple[str, list[int]]:
    if field == "sent":
        return episode["tx"]["message"], episode["tx"]["token_ids"]
    return episode["channel"]["received_message"], episode["channel"]["received_token_ids"]


def _mean(values: list[Any]) -> float:
    numeric = [float(v) for v in values if v is not None]
    return statistics.fmean(numeric) if numeric else float("nan")


def performance(episodes: list[dict[str, Any]]) -> dict[str, float]:
    row = {f"perf_{key}": _mean([e["metrics"].get(key) for e in episodes]) for key in PERFORMANCE_KEYS}
    row["perf_dominant_colour_delta_e"] = _mean([e["metrics"].get("dominant_colour_delta_e") for e in episodes])
    row["rx_valid_rate"] = _mean([e["rx"]["valid"] for e in episodes])
    row["rx_mean_attempts"] = _mean([e["rx"]["attempts"] for e in episodes])
    row["truncated_rate"] = _mean([e["channel"]["truncated"] for e in episodes])
    row["sent_mean_tokens"] = _mean([e["tx"]["token_count"] for e in episodes])
    row["mean_tx_latency_s"] = _mean([e["latency"]["tx_seconds"] for e in episodes])
    row["mean_rx_latency_s"] = _mean([e["latency"]["rx_seconds"] for e in episodes])
    judge_scores = [e["judge"]["judge_score"] for e in episodes if e.get("judge")]
    if judge_scores:
        row["judge_score"] = _mean(judge_scores)
    return row


def summarise(episodes: list[dict[str, Any]], field: str, vocab_size: int | None, n_boot: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        groups[episode["condition_id"]].append(episode)
    rows = []
    for cond, members in groups.items():
        messages, ids = zip(*(_message(e, field) for e in members))
        first = members[0]["condition"]
        rows.append({"condition": cond, "kind": "condition", **first, "n_seeds": len({e["seed"] for e in members}), **describe(list(messages), list(ids), vocab_size, n_boot), **performance(members)})
    return rows


def rounds_table(episodes: list[dict[str, Any]], field: str, vocab_size: int | None, n_boot: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        groups[(episode["condition_id"], episode["round"])].append(episode)
    if max((r for _, r in groups), default=0) == 0:
        return []
    rows = []
    for (cond, round_index), members in sorted(groups.items()):
        messages, ids = zip(*(_message(e, field) for e in members))
        rows.append({"condition": cond, "round": round_index, **members[0]["condition"], **describe(list(messages), list(ids), vocab_size, n_boot), **performance(members)})
    return rows


def baseline_rows(manifest: dict[str, Any], episodes: list[dict[str, Any]], field: str, reference: Path | None, n_messages: int, n_boot: int) -> list[dict[str, Any]]:
    cfg = manifest["config"]
    tokenizer = build_tokenizer(cfg["backend"])
    vocab_size = tokenizer.vocab_size
    rows = []
    messages, ids = template_english(cfg["dataset"]["data_dir"], tokenizer)
    rows.append({"condition": "baseline:template_english", "kind": "baseline", **describe(messages, ids, vocab_size, n_boot)})
    if reference is not None and reference.is_file():
        free = [e for e in episodes if e["condition"]["tx_mode"] == "llm_free" and e["condition"]["token_budget"] is None and not e["condition"]["memory"]]
        target = [len(_message(e, field)[1]) for e in (free or episodes)]
        messages, ids = reference_corpus(reference, tokenizer, target, n_messages, seed=0)
        rows.append({"condition": f"baseline:reference_corpus({reference.name})", "kind": "baseline", **describe(messages, ids, vocab_size, n_boot)})
    return rows


def write_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.3f}"
    return "inf" if value is None else str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path, mock: bool, field: str) -> None:
    columns = ["condition", "n_messages", "mean_tokens", "vocab_size", "type_token_ratio", "H_token_mm", "H_token_mm_boot_bc", "H_token_mm_ci_low", "H_token_mm_ci_high", "H_token_cond_mm", "H_char_mm", "H_char_cond_mm", "redundancy_token_observed", "redundancy_char_observed", "lzma_bits_per_char_net", "token_samples_per_type", "token_bigram_samples_per_type", "undersampled", "perf_attribute_score", "perf_ssim", "perf_exact_match", "rx_valid_rate"]
    lines = ["# Results summary", ""]
    if mock:
        lines += ["> **MOCK LLM RUN.** Messages and reconstructions come from the scripted mock backend. These numbers test the pipeline; they are *not* findings about LLM communication.", ""]
    lines += [f"Analysed text: **{field}** message. {CAVEAT}", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_fmt(row.get(c, "")) if c in row else "" for c in columns) + " |" for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _short(cond: str) -> str:
    return cond.replace("natural_language", "nl").replace("llm_free", "free").replace("__budget-", "\nB=").replace("__mem-", "\nmem ")


def plot_all(rows: list[dict[str, Any]], round_rows: list[dict[str, Any]], episodes: list[dict[str, Any]], out_dir: Path, mock: bool) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = "  [MOCK LLM: pipeline test, not findings]" if mock else ""
    conditions = [r for r in rows if r["kind"] == "condition"]
    baselines = [r for r in rows if r["kind"] == "baseline"]
    labels = [_short(r["condition"]) for r in conditions]
    x = list(range(len(conditions)))
    written: list[Path] = []
    colours = ["#245b8a", "#e07a5f", "#3a9d72", "#8e6bbf", "#c9a227"]

    def save(fig: Any, name: str) -> None:
        fig.tight_layout(rect=(0, 0, 1, 0.97) if fig._suptitle is not None else None)
        for ext in ("png", "pdf"):
            path = out_dir / f"{name}.{ext}"
            fig.savefig(path, dpi=150)
            written.append(path)
        plt.close(fig)

    def bar_with_ci(axis: Any, key: str, title: str) -> None:
        values = [r[key] for r in conditions]
        corrected = [r[f"{key}_boot_bc"] for r in conditions]
        lows = [c - r[f"{key}_ci_low"] for c, r in zip(corrected, conditions)]
        highs = [r[f"{key}_ci_high"] - c for c, r in zip(corrected, conditions)]
        bars = axis.bar(x, values, color=[colours[0] if not r["undersampled"] else "#9bb3c9" for r in conditions], label="Miller-Madow")
        axis.errorbar(x, corrected, yerr=[lows, highs], fmt="o", color="black", markersize=3, capsize=3, label="bootstrap bias-corrected, 95% basic CI")
        for i, baseline in enumerate(baselines):
            axis.axhline(baseline[key], color=colours[(i + 1) % len(colours)], linestyle="--", label=baseline["condition"].replace("baseline:", "English: "))
        axis.set_title(title)
        axis.set_xticks(x, labels, fontsize=7)
        axis.set_ylabel("bits / symbol")
        axis.legend(fontsize=7)
        del bars

    width = max(8, 0.55 * len(conditions))
    fig, axes = plt.subplots(2, 1, figsize=(width, 8))
    bar_with_ci(axes[0], "H_token_mm", "Unigram token entropy (light bars = undersampled, N/K<10)" + stamp)
    bar_with_ci(axes[1], "H_char_mm", "Unigram character entropy")
    save(fig, "entropy_unigram")

    fig, axes = plt.subplots(2, 1, figsize=(width, 8))
    bar_with_ci(axes[0], "H_token_cond_mm", "Conditional token entropy H(X_t | X_t-1)" + stamp)
    bar_with_ci(axes[1], "H_char_cond_mm", "Conditional character entropy H(X_t | X_t-1)")
    save(fig, "entropy_conditional")

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.8 * len(rows)), 4.5))
    all_rows = conditions + baselines
    all_labels = labels + [_short(b["condition"].replace("baseline:", "EN:")) for b in baselines]
    for axis, key, title in ((axes[0], "redundancy_token_observed", "Token redundancy 1 - H/log2(V_observed)"), (axes[1], "redundancy_char_observed", "Character redundancy 1 - H/log2(alphabet)")):
        axis.bar(range(len(all_rows)), [r[key] for r in all_rows], color=[colours[0]] * len(conditions) + [colours[1]] * len(baselines))
        axis.set_xticks(range(len(all_rows)), all_labels, fontsize=6, rotation=90)
        axis.set_title(title + stamp, fontsize=9)
    save(fig, "redundancy")

    fig, axis = plt.subplots(figsize=(max(10, 0.8 * len(rows)), 4.5))
    compressors = ("gzip", "zlib", "bz2", "lzma")
    step = 0.8 / len(compressors)
    for i, name in enumerate(compressors):
        axis.bar([j + i * step for j in range(len(all_rows))], [r[f"{name}_bits_per_char_net"] for r in all_rows], width=step, label=name, color=colours[i])
    axis.axhline(8, color="grey", linewidth=0.5)
    axis.set_xticks([j + 0.4 - step / 2 for j in range(len(all_rows))], all_labels, fontsize=6, rotation=90)
    axis.set_ylabel("compressed bits per character (header-corrected)")
    axis.set_title("Compression as entropy proxy (headers inflate small corpora)" + stamp, fontsize=9)
    axis.legend(fontsize=7)
    save(fig, "compression")

    groups: dict[str, list[list[int]]] = defaultdict(list)
    for episode in episodes:
        groups[episode["condition_id"]].append(episode["channel"]["received_token_count"])
    fig, axis = plt.subplots(figsize=(width, 4.5))
    axis.boxplot([groups[r["condition"]] for r in conditions])
    axis.set_xticks([i + 1 for i in x], labels, fontsize=7)
    axis.set_ylabel("received tokens per message")
    axis.set_title("Message length distribution" + stamp)
    save(fig, "message_lengths")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    modes = sorted({r["tx_mode"] for r in conditions})
    for axis, key, xlabel in ((axes[0], "H_token_mm", "token entropy (bits)"), (axes[1], "mean_tokens", "mean received tokens"), (axes[2], "lzma_bits_per_char_net", "lzma bits/char (net)")):
        for k, mode in enumerate(modes):
            subset = [r for r in conditions if r["tx_mode"] == mode]
            axis.scatter([r[key] for r in subset], [r["perf_attribute_score"] for r in subset], label=mode, color=colours[k % len(colours)])
        axis.set_xlabel(xlabel)
        axis.set_ylabel("attribute score")
    axes[0].legend(fontsize=7)
    fig.suptitle("Task performance vs language measures (one point per condition)" + stamp, fontsize=10)
    save(fig, "performance_vs_measures")

    budgets = sorted({r["token_budget"] for r in conditions if r["token_budget"] is not None})
    if budgets:
        ladder = budgets + [max(budgets) * 2]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for k, (mode, memory) in enumerate(sorted({(r["tx_mode"], r["memory"]) for r in conditions})):
            subset = sorted((r for r in conditions if r["tx_mode"] == mode and r["memory"] == memory), key=lambda r: ladder[-1] if r["token_budget"] is None else r["token_budget"])
            xs = [ladder[-1] if r["token_budget"] is None else r["token_budget"] for r in subset]
            style = "-o" if not memory else "--s"
            axes[0].plot(xs, [r["perf_attribute_score"] for r in subset], style, label=f"{mode}{' +mem' if memory else ''}", color=colours[k % len(colours)])
            axes[1].plot(xs, [r["H_token_mm"] for r in subset], style, color=colours[k % len(colours)])
        for axis, ylabel in ((axes[0], "attribute score"), (axes[1], "token entropy (bits)")):
            axis.set_xscale("log")
            axis.set_xticks(ladder, [str(b) for b in budgets] + ["inf"])
            axis.set_xlabel("token budget")
            axis.set_ylabel(ylabel)
        axes[0].legend(fontsize=7)
        fig.suptitle("Token-budget ladder" + stamp, fontsize=10)
        save(fig, "budget_ladder")

    if round_rows:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for k, cond in enumerate(sorted({r["condition"] for r in round_rows})):
            subset = [r for r in round_rows if r["condition"] == cond]
            for axis, key in zip(axes, ("perf_attribute_score", "mean_tokens", "H_token_mm")):
                axis.plot([r["round"] for r in subset], [r[key] for r in subset], "-o", label=_short(cond).replace("\n", " "), color=colours[k % len(colours)])
                axis.set_xlabel("round")
                axis.set_ylabel(key)
                axis.xaxis.get_major_locator().set_params(integer=True)
        axes[0].legend(fontsize=6)
        fig.suptitle("Evolution across rounds (memory = in-context adaptation)" + stamp, fontsize=10)
        save(fig, "rounds")

    examples = [e for e in episodes if e["rx"].get("reconstruction_path")][:: max(1, len(episodes) // 400)]
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for episode in examples:
        if episode["condition_id"] not in seen:
            picked.append(episode)
            seen.add(episode["condition_id"])
    picked = picked[:8]
    if picked:
        from PIL import Image

        fig, axes = plt.subplots(len(picked), 3, figsize=(10, 2.6 * len(picked)), squeeze=False, gridspec_kw={"width_ratios": [1, 1, 2]})
        for row_axes, episode in zip(axes, picked):
            for axis, path, title in ((row_axes[0], episode["image_path"], "source"), (row_axes[1], episode["rx"]["reconstruction_path"], "reconstruction")):
                try:
                    axis.imshow(Image.open(path))
                except OSError:
                    axis.text(0.5, 0.5, "missing", ha="center")
                axis.set_title(title, fontsize=8)
                axis.set_xticks([])
                axis.set_yticks([])  # keep the frame: canvases are white
            row_axes[2].axis("off")
            text = f"{episode['condition_id']}\nreceived: {episode['channel']['received_message']}\nscore={episode['metrics']['attribute_score']:.2f}  ssim={episode['metrics']['ssim']:.2f}"
            row_axes[2].text(0, 0.5, "\n".join(textwrap.wrap(text, 60, replace_whitespace=False)), fontsize=7, va="center", family="monospace")
        fig.suptitle("Examples" + stamp, fontsize=10)
        save(fig, "examples")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--field", choices=("received", "sent"), default="received")
    parser.add_argument("--reference-corpus", type=Path, help="plain-text English corpus (default from config analysis.reference_corpus)")
    parser.add_argument("--bootstrap", type=int, help="bootstrap resamples (default from config)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    episodes, manifest = load_episodes(args.run_dir)
    cfg = manifest["config"]
    n_boot = args.bootstrap if args.bootstrap is not None else int(cfg["analysis"]["bootstrap"])
    reference = args.reference_corpus or (Path(cfg["analysis"]["reference_corpus"]) if cfg["analysis"].get("reference_corpus") else None)
    vocab_size = manifest["model"].get("tokenizer_vocab_size")
    mock = manifest["model"]["backend"] == "mock"
    out_dir = args.out_dir or args.run_dir[0] / "analysis"

    rows = summarise(episodes, args.field, vocab_size, n_boot)
    rows += baseline_rows(manifest, episodes, args.field, reference, int(cfg["analysis"]["baseline_messages"]), n_boot)
    round_rows = rounds_table(episodes, args.field, vocab_size, n_boot)
    write_table(rows, out_dir / "tables" / "summary.csv")
    write_markdown(rows, out_dir / "tables" / "summary.md", mock, args.field)
    if round_rows:
        write_table(round_rows, out_dir / "tables" / "rounds.csv")
    if not args.no_plots:
        plot_all(rows, round_rows, episodes, out_dir / "figures", mock)
    print(f"Analysis written to {out_dir}")


if __name__ == "__main__":
    main()
