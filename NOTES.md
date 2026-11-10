# Design note: Rx, channel, evaluation and analysis (noiseless)

## Decisions
- **Same model for both agents.** `build_backend` loads one Qwen2.5-VL-3B-Instruct instance (the model the existing Tx targets) and passes it to Tx, Rx and, optionally, the judge. Token counts, budgets and logged token ids all use that model's tokenizer.
- **Existing Tx reused, not rewritten.** The three template encoders stay as fixed reference schemes. A new `llm_free` mode lets the LLM write its own message; this is the only mode where language can adapt.
- **Rx outputs JSON in the dataset's attribute schema, not SVG.** It is rendered with the dataset's own drawing code, so a perfect message gives a pixel-identical image. Pixel metrics then measure communication loss rather than drawing style, and attribute accuracy is exact. The cost: Rx cannot express anything outside the schema. For photos, replace the decoder (for example with SVG) behind the same `RxOutput` interface.
- **Robust Rx.** It extracts the first balanced JSON block, normalises harmless variants (`color`, `center`, `top-left`), validates every value, and retries up to `rx.max_retries` times, telling the model what was wrong. Then it falls back to the valid objects of the best attempt (`partial`) or a blank canvas (`empty`). Nothing crashes, and all raw outputs and errors are logged.
- **Order inside the channel:** Tx message → token-budget truncation (the model's tokenizer) → noise function (identity) → Rx. Tx may be told the budget (`tx.inform_budget`), but the channel enforces it in every mode.
- **Independent evaluation.** A programmatic scorer is the primary metric (SSIM, MSE, dominant-colour ΔE, Hungarian-matched attribute accuracy). An LLM judge is optional and flag-gated; with shared weights it may favour the model's own conventions.
- **Memory** is off by default and switchable, with a window of recent episodes and graded feedback. It is in-context adaptation, not training.
- **Statistics.** Entropies use the Miller–Madow correction, the bootstrap bias-corrected estimate and *basic*-bootstrap CIs over messages. Plain percentile CIs were rejected after they excluded their own estimate. N/K is reported and undersampled rows are flagged.

## Limitations
- **No real-model run was possible here** (no GPU, huggingface.co blocked). The Qwen backend follows the existing Tx's transformers usage but has **not been executed**. The committed runs use the mock, whose messages, compression under budget and "convention drift" are scripted.
- **Occlusion in the dataset.** Shapes can overlap exactly (for example `image_001`: a circle completely hidden under a square). Ground truth then contains objects a vision model cannot see. Filter or regenerate such images before real runs.
- Per-token entropy depends on the tokenizer, and comparing it across languages of different lengths is misleading; report bits per message and task score alongside it. The reference corpus is 19th-century fiction, which is off-topic; the topic-matched baselines are the English template and unconstrained `llm_free`.
- 20 images × a few seeds is small. Bigram entropies are almost always undersampled; scale the dataset (the generator allows up to 50 images) or use more seeds.
- Rx is told the attribute vocabulary, the output space. That is needed for renderable output, but it means Rx does not have to discover what the attributes are.

## How the noise channel plugs in
Register `fn(text, tokenizer, rng, **params) -> str` with `@register_noise("name")` and set `channel.noise: {name, params}` in the YAML. Nothing else changes. Budget truncation is applied before noise, the `rng` is seeded per episode, and both the sent and received messages (and their token ids) are logged, so analysis can be run on either. A **latency constraint** fits the same pattern: put a timeout into `GenerationConfig` / `max_new_tokens` and log timeouts as Rx failures through the existing fallback path.
