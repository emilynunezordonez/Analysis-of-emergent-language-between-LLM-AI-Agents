"""Small reproducible plots for aggregate baseline results."""

from pathlib import Path
from typing import Any


def plot_entropy(analysis: dict[str, Any], output_dir: str | Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Install the 'analysis' extra to create plots") from error
    output_path = Path(output_dir) / "entropy_by_mode.png"
    modes = list(analysis["modes"])
    values = [analysis["modes"][mode]["character_entropy_bits"] for mode in modes]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(modes, values, color=("#245b8a", "#e07a5f", "#3a9d72"))
    axis.set_ylabel("Empirical character entropy (bits/symbol)")
    axis.set_xlabel("Communication mode")
    axis.set_title("Pooled character entropy")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
