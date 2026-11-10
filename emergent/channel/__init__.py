"""Tx -> Rx channel: token-budget truncation followed by a pluggable noise function.

Order of operations (``Channel.transmit``):

1. **Budget**: encode with the model's own tokenizer; keep the first ``token_budget`` ids.
2. **Noise**: apply ``noise_fn(text, tokenizer, rng, **params)``. This is the
   identity in the noiseless experiment.

Adding noise later needs no refactoring. Register a function and select it in
the config::

    from emergent.channel import register_noise

    @register_noise("token_dropout")
    def token_dropout(text, tokenizer, rng, p=0.1):
        ids = [i for i in tokenizer.encode(text) if rng.random() >= p]
        return tokenizer.decode(ids)

    # configs/*.yaml:  channel: {noise: {name: token_dropout, params: {p: 0.1}}}

The channel's ``rng`` is seeded per episode, so noisy runs are reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from emergent.llm.base import Tokenizer

NoiseFn = Callable[..., str]
NOISE_REGISTRY: dict[str, NoiseFn] = {}


def register_noise(name: str) -> Callable[[NoiseFn], NoiseFn]:
    """Decorator that makes a noise function selectable by name in the config."""

    def decorator(fn: NoiseFn) -> NoiseFn:
        NOISE_REGISTRY[name] = fn
        return fn

    return decorator


@register_noise("identity")
def identity(text: str, tokenizer: Tokenizer, rng: random.Random, **params: Any) -> str:
    """Noiseless channel: the message passes through unchanged."""
    del tokenizer, rng, params
    return text


@dataclass
class ChannelOutput:
    """What went in, what came out, and the token ids on both sides."""

    sent: str
    sent_token_ids: list[int]
    after_budget: str
    received: str
    received_token_ids: list[int]
    truncated: bool
    token_budget: int | None
    noise: str


class Channel:
    """Applies the token budget, then the noise function."""

    def __init__(self, tokenizer: Tokenizer, token_budget: int | None = None, noise: str = "identity", noise_params: dict[str, Any] | None = None) -> None:
        if token_budget is not None and token_budget < 1:
            raise ValueError("token_budget must be a positive integer or None")
        if noise not in NOISE_REGISTRY:
            raise ValueError(f"unknown noise {noise!r}; registered: {sorted(NOISE_REGISTRY)}")
        self.tokenizer = tokenizer
        self.token_budget = token_budget
        self.noise = noise
        self.noise_params = dict(noise_params or {})

    def transmit(self, message: str, rng: random.Random) -> ChannelOutput:
        ids = self.tokenizer.encode(message)
        truncated = self.token_budget is not None and len(ids) > self.token_budget
        after_budget = self.tokenizer.decode(ids[: self.token_budget]) if truncated else message
        received = NOISE_REGISTRY[self.noise](after_budget, self.tokenizer, rng, **self.noise_params)
        return ChannelOutput(message, ids, after_budget, received, self.tokenizer.encode(received), truncated, self.token_budget, self.noise)
