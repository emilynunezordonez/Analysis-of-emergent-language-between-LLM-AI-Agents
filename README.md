# Information-theoretic analysis of emergent language between LLM agents (noiseless)

Two LLM agents play a reconstruction game. The **transmitter (Tx)** sees an image of simple geometric shapes and sends a text message. The **receiver (Rx)** sees *only that message* and must redraw the image. A separate, programmatic **evaluator** compares the original with the reconstruction, and every episode is logged for information-theoretic analysis.

Research questions: (1) does the language between LLM agents have lower entropy than standard English? (2) how does it change under channel noise? This repository covers the **noiseless** version, but the channel has a plug-in hook for noise (see [Adding noise](#adding-noise-later)).

> **Status of the committed results.** `runs/sample_mock/` and `runs/memory_mock/` were produced with the **mock LLM backend**, a scripted stand-in used because the development machine had no GPU. They show that the pipeline and analysis work; **they are not findings**. Every mock table and figure is stamped with this warning. Run the `configs/qwen_*.yaml` configs on a GPU machine to get real results.

## Repository layout

```
transmitter/        existing Tx package (unchanged): schema, dataset generator, template encoders, Qwen extractor
emergent/
  llm/              shared LLM backends: Qwen2.5-VL (transformers) and a deterministic mock; tokenizer wrappers
  tx/               Tx adapters: the existing template modes plus the new free-form `llm_free` mode
  rx/               Rx agent, robust JSON parsing/validation/retries, renderer
  channel/          token-budget truncation + pluggable noise function (identity for now)
  evaluation/       programmatic scorer (SSIM, MSE, dominant colour, attribute accuracy) + optional LLM judge
  memory.py         optional cross-episode memory (switchable), the mechanism for convention emergence
  runner/           config loading and the episode loop
  analysis/         entropy estimators, measures, English baselines, report + figures
configs/            YAML experiment configs (mock and Qwen)
data/               20-image dataset (seed 7) and data/reference/ English corpus
runs/               logs and analysis of the committed sample runs
tests/              pytest suite (runs without GPU or network)
NOTES.md            one-page design decisions, limitations and the noise plug-in
```

## Install

Python 3.10+ (tested on 3.13).

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
pip install -e . --no-deps
```

For the real model, on a machine with a GPU and access to huggingface.co:

```bash
pip install -r requirements-model.txt
```

## Run everything

Tests (mock LLM, about 12 s):

```bash
python -m pytest
```

Smallest end-to-end dry run (mock LLM, 4 images), then its analysis:

```bash
python -m emergent.runner.run --config configs/dry_run.yaml
```

```bash
python -m emergent.analysis.report --run-dir runs/dry_run
```

Reproduce the committed sample runs (mock LLM):

```bash
python -m emergent.runner.run --config configs/sample_mock.yaml && python -m emergent.analysis.report --run-dir runs/sample_mock
```

```bash
python -m emergent.runner.run --config configs/memory_mock.yaml && python -m emergent.analysis.report --run-dir runs/memory_mock
```

Real experiments (Qwen2.5-VL-3B-Instruct; Tx and Rx share one loaded model):

```bash
python -m emergent.runner.run --config configs/qwen_baseline.yaml
```

```bash
python -m emergent.runner.run --config configs/qwen_budget_ladder.yaml
```

```bash
python -m emergent.runner.run --config configs/qwen_memory.yaml
```

```bash
python -m emergent.analysis.report --run-dir runs/qwen_budget_ladder
```

Before a long run, a quick real-model smoke test on 2 images:

```bash
python -m emergent.runner.run --config configs/qwen_baseline.yaml --set dataset.limit=2 --seeds 0 --output-dir runs/qwen_smoke
```

### Configuration and CLI

Every option lives in a YAML file (defaults: `emergent/runner/config.py`). On the command line you can use the shortcuts `--model`, `--temperature`, `--rounds`, `--episodes`, `--seeds`, `--token-budgets inf 50 20 10 5`, `--output-dir` and `--dry-run` (forces the mock), or override any key generically:

```bash
python -m emergent.runner.run --config configs/qwen_baseline.yaml --set channel.token_budgets=[null,10] --set memory.enabled=[false,true]
```

Terms: a **condition** is (Tx mode, token budget, memory on/off). A **game** is one condition with one seed, and memory is reset at the start of each game. An **episode** is one image. `rounds` is the number of passes over the dataset per game, and `--episodes` caps the episodes per game. Each episode's seed is derived from (seed, condition, round, image), so runs are reproducible; the test suite checks this.

### Tx modes

| mode | who writes the message | can adapt? |
|---|---|---|
| `llm_free` (new) | the LLM, from the image; optionally told the budget and shown memory | yes |
| `natural_language`, `json`, `vector` | the existing deterministic templates in `transmitter/communication`, applied to attributes the VLM extracted from the image (`tx.extractor: backend`) or to ground truth (`ground_truth`, the existing oracle `MockExtractor`) | no: fixed reference schemes |

### Memory (switchable)

`memory.enabled: [false, true]` runs both settings. With memory on, Tx and Rx see the last `window` episodes in their prompts. This is in-context adaptation; no weights change. `tx_feedback` (`none | score | reconstruction`) and `rx_feedback` (`none | score | ground_truth`) control how much outcome information they get. `ground_truth` shows Rx the true attributes of *past* episodes as text. Rx never sees an image, and never sees the current episode's answer. Without memory there is nothing a convention could build on, so this switch is central to question 1. Details are in `emergent/memory.py`.

## Outputs

`runs/<name>/manifest.json` holds the resolved config, model and tokenizer, git commit and package versions. Each run directory also contains `episodes.jsonl` (one line per episode), `failures.jsonl` (Tx extraction errors and invalid Rx output, with raw text) and `reconstructions/` (PNGs, `logging.save_images` per game). `analysis/tables/summary.{csv,md}`, `analysis/tables/rounds.csv` and `analysis/figures/*.{png,pdf}` hold the analysis.

### Episode log schema (`episodes.jsonl`)

| field | content |
|---|---|
| `episode_id`, `run_id`, `condition_id`, `condition`, `seed`, `episode_seed`, `round`, `episode_index` | identification |
| `image_id`, `image_path`, `source_label`, `source_objects` | the source and its ground truth |
| `tx.message`, `tx.raw_outputs`, `tx.token_ids`, `tx.tokens`, `tx.token_count`, `tx.char_count`, `tx.word_count`, `tx.prompt`, `tx.errors`, `tx.latency_seconds` | the Tx message, tokenized with **the model's own tokenizer** |
| `channel.received_message`, `channel.received_token_ids`, `channel.received_token_count`, `channel.truncated`, `channel.token_budget`, `channel.noise` | what actually reached Rx |
| `rx.raw_outputs` (every attempt), `rx.attempts`, `rx.valid`, `rx.fallback`, `rx.errors`, `rx.reconstruction`, `rx.reconstruction_path` | Rx output, both raw and parsed |
| `metrics` | SSIM, MSE, dominant colour and ΔE, count / shape / colour / size / position accuracy, attribute score, object precision and recall, exact match |
| `judge` | optional LLM-judge score (`evaluation.llm_judge: true`) |
| `latency`, `model` (backend, model id, tokenizer, Tx and Rx decoding parameters), `memory` | run parameters |

## Analysis

`python -m emergent.analysis.report --run-dir <run>` computes these measures for each condition (and for each round when `rounds > 1`), on the *received* message by default (`--field sent` for the full Tx output):

- **Entropy:** unigram token and character entropy, and conditional (bigram) entropy H(Xₜ | Xₜ₋₁). Each uses the Miller–Madow correction plus the bootstrap bias-corrected estimate with 95% **basic**-bootstrap CIs over messages.
- **Message measures:** length distributions, vocabulary size, type–token ratio, and redundancy 1 − H/Hmax (with Hmax from the observed alphabet, and from the full tokenizer vocabulary for real models).
- **Compression:** gzip, zlib, bz2 and lzma bits per character, raw and header-corrected.
- **Performance:** task score against these measures, plus a token-budget ladder plot.
- **English baselines** tokenized with the same tokenizer: (a) the existing English template applied to the same images (same content); (b) windows from a public-domain English text (`data/reference/carroll-alice.txt`, see `SOURCE.md`) whose lengths match the unconstrained `llm_free` messages. The unconstrained `llm_free` condition itself is the model's own default English.

**Small-sample caveat.** Entropy estimates from few messages are biased low. Each row reports tokens per type, N/K, for unigrams and bigrams, and is flagged `undersampled` when N/K < 10. Don't interpret flagged rows. Unigram bits per token is not information per message: a compact code can have *higher* per-token entropy but far fewer tokens (the mock run shows exactly this). Use `est_bits_per_message_unigram`, compression bytes per message and task score together.

## Sample-run results (MOCK backend, pipeline check only)

The full table is in `runs/sample_mock/analysis/tables/summary.md`: 800 episodes covering 4 modes × 5 budgets × 2 seeds × 20 images. With the mock, the curves behave as the pipeline should. Truncation lowers the score of template modes in order of verbosity (json < natural_language < vector at a budget of 10). The budget-aware `llm_free` mock switches to codes and stays at score 1.0 down to 10 tokens. Rx retries recovered 55 of the 56 episodes that needed a retry, and 1 episode fell back to a blank canvas and was logged. The mock's behaviour is scripted, so none of this says anything about Qwen.

## Adding noise later

Register a function and select it in the config; no other code changes:

```python
from emergent.channel import register_noise

@register_noise("token_dropout")
def token_dropout(text, tokenizer, rng, p=0.1):
    return tokenizer.decode([i for i in tokenizer.encode(text) if rng.random() >= p])
```

```yaml
channel: {token_budgets: [null], noise: {name: token_dropout, params: {p: 0.1}}}
```

The channel applies the budget first, then the noise, with a per-episode seeded `rng`, and logs both the sent and the received messages.

## Original Tx baseline (unchanged)

`python -m transmitter.main --data-dir data --output-dir results --count 20 --seed 7` still runs the original template-only baseline into `results/`. Vector encoding: 13 integers `[object_count, (shape, colour, position, size) × 3]`, zero-padded; codes are shape `circle=1, square=2, triangle=3`; colour `red=1, blue=2, green=3, yellow=4`; position `centre=1, upper-left=2, upper-right=3, lower-left=4, lower-right=5`; size `small=1, medium=2, large=3`.
