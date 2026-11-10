"""Tests for Rx parsing, rendering, channel, metrics, estimators and a mock dry run."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transmitter.dataset.generator import generate_dataset
from transmitter.dataset.schema import ShapeObject

from emergent.analysis.estimators import bootstrap_ci, conditional_entropy, miller_madow_entropy, plugin_entropy, unigram_entropy
from emergent.analysis.measures import describe
from emergent.channel import NOISE_REGISTRY, Channel, register_noise
from emergent.evaluation.metrics import attribute_metrics, colour_metrics, mse, ssim
from emergent.llm.base import Generation, GenerationConfig
from emergent.llm.mock import MockBackend, MockTokenizer, mock_decode
from emergent.memory import EpisodeRecap, SharedMemory
from emergent.runner.config import load_config
from emergent.runner.run import run_experiment
from emergent.rx import RxAgent, parse_reconstruction, render

CIRCLE = {"shape": "circle", "colour": "red", "position": "centre", "size": "large"}
TRIANGLE = {"shape": "triangle", "colour": "blue", "position": "upper-left", "size": "small"}


# ---------- Rx output parsing ----------

def test_parse_plain_json():
    result = parse_reconstruction(json.dumps({"objects": [CIRCLE, TRIANGLE]}))
    assert result.valid and result.objects_as_dicts() == [CIRCLE, TRIANGLE]


def test_parse_code_fence_prose_and_synonyms():
    text = 'Sure! Here it is:\n```json\n{"objects": [{"type": "Circle", "color": "RED", "position": "center", "size": "big"}]}\n```\nHope that helps {really}.'
    result = parse_reconstruction(text)
    assert result.valid and result.objects_as_dicts() == [CIRCLE]


def test_parse_top_level_list_and_bottom_right():
    result = parse_reconstruction('[{"shape":"square","colour":"green","position":"bottom right","size":"medium"}]')
    assert result.valid and result.objects[0].position == "lower-right"


def test_parse_braces_inside_strings():
    result = parse_reconstruction('{"note": "a } tricky {", "objects": [' + json.dumps(CIRCLE) + "]}")
    assert result.valid and len(result.objects) == 1


@pytest.mark.parametrize("text", ["no json here", '{"objects": [{"shape": "circle"', "{'objects': []}", '{"shapes": []}'])
def test_parse_rejects_malformed(text):
    assert not parse_reconstruction(text).valid


def test_parse_invalid_values_keep_valid_objects():
    bad = {**TRIANGLE, "shape": "hexagon"}
    result = parse_reconstruction(json.dumps({"objects": [CIRCLE, bad]}))
    assert not result.valid
    assert result.objects_as_dicts() == [CIRCLE]
    assert any("hexagon" in e for e in result.errors)


def test_parse_too_many_objects():
    result = parse_reconstruction(json.dumps({"objects": [CIRCLE] * 5}), max_objects=3)
    assert not result.valid and len(result.objects) == 3


# ---------- Rx agent: retries and failure logging ----------

class ScriptedBackend:
    """Returns pre-set outputs in order; records what it was shown."""

    model_id = "scripted"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.tokenizer = MockTokenizer()
        self.calls = []

    def generate(self, messages, config, *, task, hints=None):
        self.calls.append(messages)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return Generation(output, 0.01)


def test_rx_retries_then_succeeds():
    backend = ScriptedBackend(["garbage", json.dumps({"objects": [CIRCLE]})])
    out = RxAgent(backend, max_retries=2).receive("big red circle", memory=SharedMemory(), generation=GenerationConfig(seed=1))
    assert out.valid and out.attempts == 2 and out.objects == [CIRCLE] and len(out.errors) == 1
    assert "invalid" in backend.calls[1][-1].content  # the retry tells the model what was wrong


def test_rx_never_crashes_and_falls_back():
    backend = ScriptedBackend(["nope", RuntimeError("CUDA OOM"), json.dumps({"objects": [CIRCLE, {**CIRCLE, "colour": "pink"}]})])
    out = RxAgent(backend, max_retries=2).receive("msg", memory=SharedMemory(), generation=GenerationConfig())
    assert not out.valid and out.fallback == "partial" and out.objects == [CIRCLE]
    assert any("CUDA OOM" in e for e in out.errors)
    empty = RxAgent(ScriptedBackend(["x", "y", "z"]), max_retries=2).receive("msg", memory=SharedMemory(), generation=GenerationConfig())
    assert empty.fallback == "empty" and empty.objects == [] and empty.attempts == 3


def test_rx_prompt_contains_only_message_and_memory():
    backend = ScriptedBackend([json.dumps({"objects": [CIRCLE]})])
    RxAgent(backend).receive("THE MESSAGE", memory=SharedMemory(), generation=GenerationConfig())
    messages = backend.calls[0]
    assert all(not m.images for m in messages)  # Rx never sees an image
    assert "THE MESSAGE" in messages[-1].content


# ---------- Rendering ----------

def test_render_matches_dataset_generator(tmp_path):
    attributes = generate_dataset(tmp_path, count=3, seed=11)
    for attrs in attributes:
        source = Image.open(tmp_path / "images" / f"{attrs.image_id}.png").convert("RGB")
        assert np.array_equal(np.asarray(render(attrs.objects)), np.asarray(source))


def test_render_size_and_blank():
    assert render([], (128, 128)).size == (128, 128)
    assert (np.asarray(render([])) == 255).all()


# ---------- Metrics ----------

def test_metrics_identical_images():
    image = render([ShapeObject(**CIRCLE)])
    assert ssim(image, image) == pytest.approx(1.0)
    assert mse(image, image) == 0.0
    colour = colour_metrics(image, image)
    assert colour["dominant_colour_delta_e"] == 0.0 and colour["dominant_colour_match"]
    attrs = attribute_metrics([CIRCLE], [CIRCLE])
    assert attrs["exact_match"] and attrs["attribute_score"] == 1.0 and attrs["shape_accuracy"] == 1.0


def test_metrics_different_images():
    a, b = render([ShapeObject(**CIRCLE)]), render([ShapeObject(**TRIANGLE)])
    assert ssim(a, b) < 0.9 and mse(a, b) > 0
    colour = colour_metrics(a, b)
    assert colour["dominant_colour_delta_e"] > 10 and not colour["dominant_colour_match"]
    attrs = attribute_metrics([CIRCLE], [TRIANGLE])
    assert not attrs["exact_match"] and attrs["shape_accuracy"] == 0.0 and attrs["colour_accuracy"] == 0.0


def test_attribute_matching_is_order_invariant_and_penalises_extras():
    assert attribute_metrics([CIRCLE, TRIANGLE], [TRIANGLE, CIRCLE])["exact_match"]
    extra = attribute_metrics([CIRCLE], [CIRCLE, TRIANGLE])
    assert extra["object_recall"] == 1.0 and extra["object_precision"] == 0.5 and extra["attribute_score"] == 0.5
    blank = colour_metrics(render([ShapeObject(**CIRCLE)]), render([]))
    assert blank["dominant_colour_delta_e"] is None and not blank["dominant_colour_match"]


# ---------- Channel ----------

def test_channel_identity_and_budget():
    tokenizer = MockTokenizer()
    message = "a large red circle in the centre"
    out = Channel(tokenizer).transmit(message, random.Random(0))
    assert out.received == message and not out.truncated
    cut = Channel(tokenizer, token_budget=3).transmit(message, random.Random(0))
    assert cut.truncated and len(cut.received_token_ids) == 3 and message.startswith(cut.received)


def test_channel_noise_plugs_in_without_changes():
    @register_noise("test_upper")
    def upper(text, tokenizer, rng, suffix=""):
        return text.upper() + suffix

    try:
        out = Channel(MockTokenizer(), noise="test_upper", noise_params={"suffix": "!"}).transmit("abc", random.Random(0))
        assert out.received == "ABC!" and out.sent == "abc"
    finally:
        NOISE_REGISTRY.pop("test_upper")
    with pytest.raises(ValueError):
        Channel(MockTokenizer(), noise="missing")


def test_mock_tokenizer_roundtrip():
    tokenizer = MockTokenizer()
    for text in ["", "  héllo_wörld\t\n x", "[3,1,1,4]", "MRcLL LBcC"]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


# ---------- Memory ----------

def test_memory_views_and_window():
    memory = SharedMemory(enabled=True, window=2, tx_feedback="score", rx_feedback="ground_truth")
    for i in range(3):
        memory.add(EpisodeRecap(i, f"msg{i}", f"msg{i}", [CIRCLE], [TRIANGLE], 0.5))
    assert len(memory) == 2 and "msg0" not in memory.tx_view() and "score 0.50" in memory.tx_view()
    assert "correct answer" in memory.rx_view()
    assert SharedMemory(enabled=False).tx_view() is None


# ---------- Mock decoder ----------

def test_mock_decoder_handles_all_formats():
    assert mock_decode("LBsLL SRcLR") == [{"shape": "square", "colour": "blue", "position": "lower-left", "size": "large"}, {"shape": "circle", "colour": "red", "position": "lower-right", "size": "small"}]
    assert mock_decode("[1,2,2,4,3,0,0,0,0,0,0,0,0]")[0]["shape"] == "square"
    assert mock_decode('{"objects":[{"shape":"triangle","colour":"green"')[0]["shape"] == "triangle"
    assert mock_decode("A small yellow triangle sits in the top-left corner.")[0] == {"shape": "triangle", "colour": "yellow", "position": "upper-left", "size": "small"}


# ---------- Estimators ----------

def test_entropy_estimators():
    uniform = Counter({c: 10 for c in "abcd"})
    assert plugin_entropy(uniform) == pytest.approx(2.0)
    assert miller_madow_entropy(uniform) > 2.0
    assert unigram_entropy([["a"] * 5], "plugin") == 0.0
    assert conditional_entropy([list("abababab")], "plugin") < 0.5  # predictable sequence
    sequences = [list("abcd"), list("aabb"), list("abca")] * 5
    low, high, corrected = bootstrap_ci(sequences, unigram_entropy, n_boot=200, seed=0)
    assert low <= corrected <= high and low <= unigram_entropy(sequences) + 0.2


def test_describe_outputs_core_measures():
    tokenizer = MockTokenizer()
    messages = ["a red circle", "a blue square", "a red square"]
    row = describe(messages, [tokenizer.encode(m) for m in messages], None, n_boot=50)
    for key in ("H_token_mm", "H_char_cond_mm", "type_token_ratio", "redundancy_char_observed", "lzma_bits_per_char_net", "undersampled"):
        assert key in row
    assert 0 <= row["redundancy_token_observed"] <= 1


# ---------- End-to-end dry run with the mock LLM ----------

def test_dry_run_end_to_end(tmp_path):
    generate_dataset(tmp_path / "data", count=3, seed=7)
    cfg = load_config(None, [
        f"dataset.data_dir={tmp_path / 'data'}", f"experiment.output_dir={tmp_path / 'run'}", "experiment.rounds=2",
        "tx.modes=[llm_free, vector]", "channel.token_budgets=[null, 5]", "memory.enabled=[false, true]",
        "backend.mock.malformed_rate=0.3", "evaluation.llm_judge=true", "analysis.bootstrap=20",
    ])
    out_dir = run_experiment(cfg, MockBackend(malformed_rate=0.3))
    episodes = [json.loads(line) for line in (out_dir / "episodes.jsonl").read_text().splitlines()]
    assert len(episodes) == 2 * 2 * 2 * 3 * 2  # modes x budgets x memory x images x rounds
    first = episodes[0]
    for key in ("episode_id", "seed", "source_label", "tx", "channel", "rx", "metrics", "latency", "model", "judge"):
        assert key in first
    assert first["tx"]["token_count"] == len(first["tx"]["token_ids"]) and first["tx"]["char_count"] == len(first["tx"]["message"])
    assert all(e["channel"]["received_token_count"] <= 5 for e in episodes if e["condition"]["token_budget"] == 5)
    assert any(e["memory"]["size_at_start"] > 0 for e in episodes if e["condition"]["memory"])
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["n_episodes"] == len(episodes) and manifest["model"]["backend"] == "mock"

    # reproducibility: the same config yields the same messages and metrics
    again = run_experiment({**cfg, "experiment": {**cfg["experiment"], "output_dir": str(tmp_path / "run2")}}, MockBackend(malformed_rate=0.3))
    episodes2 = [json.loads(line) for line in (again / "episodes.jsonl").read_text().splitlines()]
    strip = lambda e: (e["episode_id"], e["tx"]["message"], e["channel"]["received_message"], e["rx"]["raw_outputs"], e["metrics"]["ssim"])
    assert [strip(e) for e in episodes] == [strip(e) for e in episodes2]

    from emergent.analysis.report import main as report_main

    report_main(["--run-dir", str(out_dir), "--bootstrap", "20"])
    summary = (out_dir / "analysis" / "tables" / "summary.csv").read_text()
    assert "baseline:template_english" in summary and "llm_free__budget-5__mem-on" in summary
    assert (out_dir / "analysis" / "figures" / "entropy_unigram.png").exists()
    assert (out_dir / "analysis" / "tables" / "rounds.csv").exists()
    assert not math.isnan(float(summary.splitlines()[1].split(",")[summary.splitlines()[0].split(",").index("H_token_mm")]))
