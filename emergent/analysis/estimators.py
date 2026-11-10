"""Entropy estimators with small-sample bias correction and bootstrap CIs.

CAVEAT: the plug-in (maximum-likelihood) entropy estimator is biased downwards
when the number of observations N is not much larger than the number of
symbol types K. Unseen symbols are invisible and frequencies of rare symbols
are noisy. Message sets from a few dozen episodes are exactly in that regime,
especially for token unigrams and, even more so, bigrams. We therefore report:

* the Miller-Madow corrected estimate  H_MM = H_plugin + (K - 1) / (2 N ln 2);
* *basic* (reverse-percentile) bootstrap confidence intervals that resample
  *messages* (not tokens), because tokens inside a message are not independent.
  Plain percentile intervals are wrong here: duplicated messages in a resample
  hide types, so bootstrap entropies are biased low and the percentile interval
  can even exclude the point estimate. The basic interval
  [2*H - q_hi, 2*H - q_lo] reflects that bias back. It is centred on the
  bootstrap bias-corrected estimate 2*H - mean(H*), which is also reported.
  When that correction is large, the CI may not contain the Miller-Madow value:
  that is a warning that the sample is too small;
* ``N / K`` alongside each estimate. Treat values with N/K < 10 as unreliable.

Miller-Madow only corrects the first-order bias. For strong claims, also
compare at equal sample sizes, or use NSB / Chao-Shen estimators.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Callable, Hashable, Iterable, Sequence

BOS = "<s>"


def plugin_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c)


def miller_madow_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    observed_types = sum(1 for c in counts.values() if c)
    return plugin_entropy(counts) + (observed_types - 1) / (2 * total * math.log(2))


ESTIMATORS: dict[str, Callable[[Counter], float]] = {"plugin": plugin_entropy, "miller_madow": miller_madow_entropy}


def unigram_counts(sequences: Iterable[Sequence[Hashable]]) -> Counter:
    return Counter(symbol for sequence in sequences for symbol in sequence)


def unigram_entropy(sequences: Sequence[Sequence[Hashable]], estimator: str = "miller_madow") -> float:
    """Entropy (bits/symbol) of the pooled unigram distribution."""
    return ESTIMATORS[estimator](unigram_counts(sequences))


def conditional_entropy(sequences: Sequence[Sequence[Hashable]], estimator: str = "miller_madow") -> float:
    """H(X_t | X_{t-1}) = H(X_{t-1}, X_t) - H(X_{t-1}), with a start symbol per message.

    Needs far more data than the unigram entropy; check ``N / K`` for bigrams.
    """
    pairs: Counter = Counter()
    contexts: Counter = Counter()
    for sequence in sequences:
        previous: Hashable = BOS
        for symbol in sequence:
            pairs[(previous, symbol)] += 1
            contexts[previous] += 1
            previous = symbol
    estimate = ESTIMATORS[estimator]
    return max(0.0, estimate(pairs) - estimate(contexts))


def bootstrap_ci(sequences: Sequence[Sequence[Hashable]], statistic: Callable[[Sequence[Sequence[Hashable]]], float], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Basic bootstrap CI (resampling whole messages), clipped at 0.

    Returns ``(low, high, bias_corrected_estimate)``.
    """
    if not sequences or n_boot <= 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(sequences)
    values = sorted(statistic([sequences[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = values[int(math.floor(alpha / 2 * (n_boot - 1)))]
    hi = values[int(math.ceil((1 - alpha / 2) * (n_boot - 1)))]
    estimate = statistic(sequences)
    corrected = 2 * estimate - sum(values) / n_boot
    return (max(0.0, 2 * estimate - hi), max(0.0, 2 * estimate - lo), max(0.0, corrected))
