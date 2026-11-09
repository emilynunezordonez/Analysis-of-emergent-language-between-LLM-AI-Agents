---
name: information-theoretic-transmitter
description: "Use when designing, implementing, testing, or reviewing the noiseless Transmitter (Tx) experiment for information-theoretic analysis of communication between LLM agents. Covers geometric-image datasets, Qwen2.5-VL-style vision-language inference, natural-language/JSON/vector encodings, receiver-facing message contracts, entropy and message metrics, reproducible result storage, and staged validation."
---

# Information-Theoretic Transmitter

Build the Tx component as a reproducible baseline experiment. The component receives an image of geometric objects, extracts the same visual attributes into three predefined message formats, measures those messages, and exposes them unchanged to an independent Receiver (Rx).

This skill applies only to the initial noiseless experiment. Do not delete, substitute, randomly alter, or otherwise transform messages after generation. Do not implement the Rx agent unless explicitly requested.

## Research boundaries

- Treat natural language, JSON, and numerical vectors as predefined communication schemes, not evidence of emergent language.
- Keep the source image attributes and the semantic content identical across all three modes.
- Distinguish empirical entropy of the collected message dataset from conditional entropy and entropy rate. Do not label one as another.
- Record the tokenizer and tokenizer configuration used for token-level metrics.
- Preserve the original generated message exactly; derived metrics and normalized analysis data must be stored separately.
- Keep model, prompt, dataset, decoding settings, and execution environment reproducible and report them with the results.

## Workflow

### 1. Establish the experiment contract

Before writing implementation code, document:

- The image attribute schema: shape (`circle`, `square`, `triangle`), colour (`red`, `blue`, `green`, `yellow`), position (`centre`, `upper-left`, `upper-right`, `lower-left`, `lower-right`), size (`small`, `medium`, `large`), and object count.
- The dataset identifier, image IDs, expected attributes, and dataset split or seed.
- The model identifier, processor/tokenizer, prompt, device, precision, decoding parameters, and software versions.
- The exact schemas for the three messages and the Tx-to-Rx interface.
- The unit and scope of every metric: per message versus aggregate dataset, characters versus tokens, and empirical entropy versus other information-theoretic quantities.

If a required decision is missing, mark it as an explicit configuration value or open research question. Do not silently invent incompatible schemas.

### 2. Build and verify the dataset

Create a small initial dataset of approximately 20-50 images containing one or more geometric objects. Store ground-truth attributes beside each image, using stable image IDs.

Check independently that:

- every image is loadable;
- every declared attribute is in the allowed vocabulary;
- multiple-object images have deterministic object ordering;
- the count in metadata matches the objects in the image;
- no image or metadata row is duplicated unintentionally.

Use a deterministic seed for generated examples and save the generation configuration.

### 3. Implement the vision extraction boundary

Create one model adapter that accepts an image and returns a validated internal representation of the image attributes. Keep model loading, preprocessing, prompting, inference, and timing inside the adapter or an equivalent model module.

The adapter must:

- use one configured open-source vision-language model, such as Qwen2.5-VL;
- request machine-readable attributes rather than three independently improvised descriptions;
- validate shape, colour, position, size, object count, and object ordering;
- fail clearly on malformed or incomplete output;
- record generation time and relevant model settings.

Derive all three communication modes from the validated internal representation wherever possible. This ensures that differences in measured entropy reflect the encoding, not unequal visual interpretation.

### 4. Implement the three encoders

Keep each mode in its own module and give each encoder a stable, documented contract.

#### Mode A: Natural language

Generate a deterministic ordinary-English description containing exactly the objects and attributes in the internal representation. Define ordering, punctuation, number words, and whether a fixed template is used. Do not add scene details that are absent from the schema.

#### Mode B: Structured JSON

Serialize the same representation using a documented JSON schema. Define key names, object ordering, whitespace policy, Unicode policy, and whether serialization is canonical. Store the exact serialized string sent to Rx.

#### Mode C: Numerical vector

Define and document every vector position and encoding. Use a representation that supports multiple objects without losing count or order, for example a fixed-size vector with an object-count field and one fixed block per possible object, or a separately documented sequence representation. Define vocabulary-to-number mappings, position encoding, padding, and numeric precision.

Reject a vector design that cannot represent every image in the dataset or that omits an attribute represented by another mode.

### 5. Define the Receiver-facing interface

Expose a small interface that accepts a message plus its mode and metadata. The interface should transport the exact message unchanged and should not parse or reinterpret it on the Tx side beyond validation needed at generation time.

At minimum, specify:

- mode identifier;
- image ID and experiment ID;
- exact message payload;
- schema or encoding version;
- timestamp or generation record;
- optional source attributes for evaluation, kept separate from the payload.

Provide a local transport or function-level adapter suitable for an independent Rx implementation. Do not couple the interface to Rx internals.

### 6. Measure every message consistently

For each image and each mode, record:

- image ID and communication mode;
- exact generated message;
- character count;
- word count using a documented rule;
- token count using the recorded tokenizer;
- character-level and token-level frequency information;
- generation time and model timing where available;
- original image attributes or a reference to the ground truth.

Use the same counting policy across modes where the data type permits it. Document edge cases such as JSON punctuation, vector delimiters, whitespace, and special tokenizer tokens.

### 7. Compute aggregate information-theoretic results

Calculate metrics from the complete collected message datasets and retain the analysis inputs:

- empirical character-level Shannon entropy;
- empirical token-level Shannon entropy;
- token frequency distributions;
- message length distributions;
- redundancy and compression measures, with definitions and compressor settings;
- task or reconstruction performance when an Rx evaluation is available.

State the sample space and aggregation unit for each entropy value. A dataset-level empirical entropy is not automatically an entropy rate. Avoid comparing values produced from incompatible tokenizers, corpora, or preprocessing rules.

Save machine-readable results in CSV or JSON and produce plots with labels, units, sample sizes, and experiment configuration references.

### 8. Integrate only after component checks pass

Integrate the dataset loader, model adapter, validators, encoders, receiver interface, metrics, and result writer through a small orchestration entry point such as `main.py`.

The integrated run must:

1. load a configured dataset;
2. infer and validate attributes;
3. generate all three exact messages;
4. measure and persist each record;
5. calculate aggregate metrics;
6. save configuration and provenance;
7. report failures without silently fabricating messages.

Support a dry-run or mock model path so schema, encoding, metrics, and result persistence can be tested without downloading a large model.

## Suggested project layout

```text
transmitter/
  main.py
  model/
    adapter.py
  dataset/
    loader.py
    schema.py
  prompts/
    visual_extraction.txt
  communication/
    natural_language.py
    json_format.py
    vector_format.py
    interface.py
  analysis/
    entropy.py
    message_metrics.py
    distributions.py
  results/
    writer.py
    plots.py
tests/
  test_dataset.py
  test_encoders.py
  test_metrics.py
  test_interface.py
configs/
  experiment.json
  vector_encoding.json
data/
  images/
  metadata/
results/
```

Adapt the layout to the existing repository rather than creating duplicate abstractions.

## Decision points

- **Model output versus deterministic encoders:** prefer one validated attribute extraction followed by deterministic encoders; use separate model generations only when the experiment explicitly tests generation differences, and document that confounder.
- **Vector shape:** choose fixed-size blocks when the dataset has a known maximum object count; choose a length-prefixed sequence when arbitrary counts are required. In either case, publish the mapping and padding rules.
- **Tokenization:** use the target model tokenizer for token-level results and record its identifier. If comparing modes across tokenizers, add a separate explicitly labeled comparison rather than mixing token counts.
- **Canonical JSON:** use canonical serialization for reproducible baseline measurements; preserve a second raw model output only if the experiment needs to study formatting variation.
- **Ground truth disagreement:** retain both ground truth and model-extracted attributes, mark the record as invalid or mismatched, and report the exclusion rule. Never silently replace model output with ground truth in the communication payload.
- **Failed generation:** record the failure and configuration, then either stop the run or apply a predeclared exclusion policy. Never insert an empty or guessed message.

## Required validation

Run component tests before model integration:

- dataset schema and deterministic ordering tests;
- encoder equivalence tests showing that all modes preserve the same attributes;
- JSON parse and schema validation tests;
- vector round-trip tests for one and multiple objects;
- message immutability tests for the Receiver interface;
- character, word, and tokenizer-based token count tests;
- entropy tests against small hand-computed examples;
- result writer tests proving exact messages survive serialization;
- mock end-to-end run covering all three modes.

Then run the smallest real-model smoke test available and compare its output against the schema. For the initial 20-50 image experiment, verify that every result row has the required fields, every mode has the same image coverage, and the provenance configuration is present.

## Completion criteria

The Tx baseline is complete only when:

- all three encodings are documented and represent identical source attributes;
- messages are transmitted and stored unchanged;
- the Rx interface is modular and independent of Rx implementation;
- character and token metrics use documented, reproducible rules;
- aggregate entropy and frequency results identify their sample space and tokenizer;
- timing, configuration, model, dataset, and failures are reproducible from saved artifacts;
- tests pass for isolated components and the mock integrated pipeline;
- the real-model smoke test produces inspectable outputs without unsupported claims about emergent language.

## Expected outputs

Produce:

- modular Python source and focused tests;
- installation and execution instructions;
- a small example dataset or deterministic dataset generator;
- per-message CSV or JSON records;
- aggregate metric files and visualizations;
- an explanation of the three communication modes, vector mapping, tokenizer, and entropy definitions;
- a short limitations note stating that predefined JSON and vector schemes are baselines, not emergent languages.

## Example prompts to use this skill

- "Implement the noiseless Tx baseline from this skill, starting with the dataset schema and encoder tests."
- "Review the transmitter results for entropy-methodology errors, tokenizer inconsistencies, and message mutations."
- "Add a deterministic vector encoding that supports up to three objects and prove it with round-trip tests."
- "Run the mock pipeline and generate per-message metrics and aggregate Shannon entropy without loading the real model."
- "Prepare the real Qwen2.5-VL smoke test while preserving the existing result schema and provenance."
