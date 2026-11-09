# Information-Theoretic Transmitter

Noiseless Tx baseline for comparing three predefined representations of geometric images. This project does not claim that JSON or vectors are emergent language.

## Architecture

`dataset` validates image attributes and can generate a deterministic 20-image PNG dataset. `model` owns visual extraction; the default `MockExtractor` uses stored ground truth so tests do not require a GPU. `communication` derives natural language, canonical JSON, and a fixed-size vector from the same validated attributes. `analysis` measures characters, whitespace-delimited words, regex word/punctuation tokens, empirical Shannon entropy, and token frequencies. `results` preserves exact payloads in CSV and JSON.

## Vector encoding

A vector has 13 integers: `[object_count, shape, colour, position, size, ...]`, followed by zero padding to three object blocks. Codes are shape `circle=1, square=2, triangle=3`; colour `red=1, blue=2, green=3, yellow=4`; position `centre=1, upper-left=2, upper-right=3, lower-left=4, lower-right=5`; size `small=1, medium=2, large=3`. Object order is the metadata order.

## Install and run

```powershell
python -m pip install -e ".[test,analysis]"
pytest
python -m transmitter.main --data-dir data --output-dir results --count 20 --seed 7
```

The run writes `data/images`, `data/metadata.json`, `results/messages.csv`, `results/messages.json`, and `results/analysis.json`. Token-level metrics use `regex-word-punctuation-v1`; a real-model study must record the Qwen tokenizer identifier instead.

## Real VLM boundary

Install the optional model dependencies with `python -m pip install -e ".[model]"`. The `QwenVisionExtractor` is intentionally an explicit integration boundary and currently raises `NotImplementedError` until schema-constrained Qwen JSON parsing is configured. This prevents silently fabricating or mutating messages in the baseline.

## Research limits

The reported entropy is empirical Shannon entropy over pooled characters or tokens in the collected messages. It is not conditional entropy or entropy rate. The Receiver is not implemented; `MessageEnvelope` transports the exact payload plus experiment metadata for an independent Rx implementation.
