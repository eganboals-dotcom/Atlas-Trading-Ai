"""v37: no training target reaches the rebalance day, the AR(1) forecast, and the features' shape."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research" / "investing"))
import v37_forecasters as v  # noqa: E402


def test_training_targets_all_end_before_the_rebalance() -> None:
    first, t = 1003, 2000
    dates = v.training_dates(t, first)
    assert dates and all(s + v.H - 1 < t for s in dates)  # the target's last close is row s + H - 1
    assert all((s - first) % v.cfg["every"] == 0 for s in dates)
    assert min(dates) >= t - v.cfg["train_years"] * 252


def test_ar1_follows_a_known_autoregression() -> None:
    steps = np.empty(253)
    steps[0] = 0.01
    for i in range(1, 253):
        steps[i] = 0.002 + 0.5 * steps[i - 1]  # no noise: phi 0.5, c 0.002, settling at 0.004
    logp = np.concatenate([[0.0], np.cumsum(steps)])[:, None].repeat(2, axis=1)
    assert v.ar1(logp, len(logp)) == pytest.approx(np.full(2, 0.004 * v.H), rel=1e-6)


def test_features_and_ranks_have_one_row_per_company_in_0_to_1() -> None:
    rng = np.random.default_rng(0)
    logp = np.cumsum(rng.normal(0, 0.01, (300, 7)), axis=0)
    f = v.features(logp, np.exp(rng.normal(10, 1, (300, 7))), 300)
    assert f.shape == (7, 10) and f.min() == 0.0 and f.max() == 1.0
    assert v.target(logp, 290).shape == (7,)
