"""Run the noiseless Tx baseline with deterministic mock extraction."""

import argparse
import json
from pathlib import Path

from transmitter.analysis.entropy import character_entropy, frequency_distribution, token_entropy
from transmitter.analysis.message_metrics import measure
from transmitter.analysis.plots import plot_entropy
from transmitter.communication.interface import MessageEnvelope
from transmitter.communication import json_format, natural_language, vector_format
from transmitter.dataset.generator import generate_dataset
from transmitter.dataset.loader import load_dataset
from transmitter.model.adapter import MockExtractor
from transmitter.results.writer import write_analysis, write_records


def run(data_dir: str | Path, output_dir: str | Path, count: int = 20, seed: int = 7, experiment_id: str = "tx-baseline-001") -> dict:
    data_path = Path(data_dir)
    if not (data_path / "metadata.json").exists():
        generate_dataset(data_path, count=count, seed=seed)
    dataset = load_dataset(data_path)
    extractor = MockExtractor()
    encoders = {"natural_language": natural_language.encode, "json": json_format.encode, "vector": lambda attrs: json.dumps(vector_format.encode(attrs), separators=(",", ":"))}
    records: list[dict] = []
    by_mode: dict[str, list[str]] = {mode: [] for mode in encoders}
    token_sequences: dict[str, list[tuple[str, ...]]] = {mode: [] for mode in encoders}
    for image_path, ground_truth in dataset:
        extracted, generation_time = extractor.extract(image_path, ground_truth)
        for mode, encoder in encoders.items():
            payload = encoder(extracted)
            metrics = measure(payload)
            envelope = MessageEnvelope(experiment_id, extracted.image_id, mode, payload)
            by_mode[mode].append(payload)
            token_sequences[mode].append(metrics.tokens)
            records.append({**envelope.as_dict(), "message": payload, "character_count": metrics.character_count, "word_count": metrics.word_count, "token_count": metrics.token_count, "generation_time_seconds": generation_time, "tokenizer": metrics.tokenizer, "original_attributes": json.dumps(ground_truth.to_dict(), ensure_ascii=False)})
    analysis = {"experiment_id": experiment_id, "model": extractor.model_name, "entropy_definition": "empirical Shannon entropy over pooled characters or tokens; not entropy rate", "modes": {mode: {"message_count": len(messages), "character_entropy_bits": character_entropy(messages), "token_entropy_bits": token_entropy(token_sequences[mode]), "token_frequency": frequency_distribution(token for sequence in token_sequences[mode] for token in sequence)} for mode, messages in by_mode.items()}}
    write_records(records, output_dir)
    write_analysis(analysis, output_dir)
    try:
        plot_entropy(analysis, output_dir)
    except (ImportError, MemoryError):
        pass
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.count, args.seed)
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
