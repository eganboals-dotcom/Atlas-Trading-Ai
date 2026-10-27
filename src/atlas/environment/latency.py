"""Seeded latency sampling. The same seed always yields the same latency sequence."""

from __future__ import annotations

import math

import numpy as np

from atlas.config import LatencyDistribution
from atlas.timeutils import NS_PER_MS


class LatencySampler:
    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def sample_ns(self, dist: LatencyDistribution) -> int:
        if dist.kind == "fixed":
            ms = dist.value_ms
        elif dist.kind == "uniform":
            ms = float(self._rng.uniform(dist.low_ms, dist.high_ms))
        else:  # lognormal with median value_ms
            ms = float(self._rng.lognormal(math.log(dist.value_ms), dist.sigma))
        return max(0, int(round(ms * NS_PER_MS)))
