"""Standard-English baselines for research question 1, tokenized with the run's tokenizer.

* ``template_english``: the existing Tx natural-language template applied to the
  same images. This is plain English with the same content, but tiny lexical variety.
* ``reference_corpus``: contiguous windows from a public-domain English text
  (default: *Alice's Adventures in Wonderland*, from NLTK's Gutenberg selection).
  Window lengths are resampled from the token lengths of the messages being
  compared, so both sides have comparable message lengths and sample sizes.

A third, implicit baseline is the ``llm_free`` condition with no budget and no
memory: the model's own default English for this task.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Sequence

from transmitter.communication import natural_language
from transmitter.dataset.loader import load_dataset

from emergent.llm.base import Tokenizer


def template_english(data_dir: str | Path, tokenizer: Tokenizer) -> tuple[list[str], list[list[int]]]:
    messages = [natural_language.encode(attributes) for _, attributes in load_dataset(data_dir)]
    return messages, [tokenizer.encode(m) for m in messages]


def reference_corpus(path: str | Path, tokenizer: Tokenizer, target_lengths: Sequence[int], n_messages: int, seed: int = 0) -> tuple[list[str], list[list[int]]]:
    """Sample ``n_messages`` token windows whose lengths follow ``target_lengths``."""
    text = re.sub(r"\s+", " ", Path(path).read_text(encoding="utf-8")).strip()
    ids = tokenizer.encode(text)
    lengths = [max(1, int(n)) for n in target_lengths] or [20]
    rng = random.Random(seed)
    messages, sequences = [], []
    for _ in range(n_messages):
        length = min(rng.choice(lengths), len(ids))
        start = rng.randrange(0, len(ids) - length + 1)
        window = ids[start: start + length]
        messages.append(tokenizer.decode(window))
        sequences.append(window)
    return messages, sequences
