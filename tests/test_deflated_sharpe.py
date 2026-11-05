"""The Deflated Sharpe formula: the best of many useless rules scores high, and more tries make any result harder to believe."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "research" / "investing"))
import v26_deflated_sharpe as dsr  # noqa: E402


def test_expected_best_matches_the_published_value():
    # Bailey and Lopez de Prado: the best of 1,000 useless rules reaches about 3.26 standard deviations.
    assert dsr.expected_best(1000, 1.0) == pytest.approx(3.26, abs=0.01)


def test_expected_best_scales_with_the_spread_of_the_trials():
    assert dsr.expected_best(25, 0.04) == pytest.approx(0.2 * dsr.expected_best(25, 1.0))


def test_one_try_has_nothing_to_deflate():
    assert dsr.expected_best(1, 1.0) == 0.0


def test_more_tries_lower_the_answer():
    r = np.random.default_rng(1).normal(0.001, 0.01, 1500)
    answers = [dsr.deflated(r, n, 0.1 / 252) for n in (1, 10, 100, 1000)]
    assert answers == sorted(answers, reverse=True)
    assert 0.0 < answers[-1] < answers[0] < 1.0


def test_no_edge_scores_near_a_half_with_no_deflating():
    r = np.random.default_rng(2).normal(0.0, 0.01, 200_000)
    assert dsr.deflated(r - r.mean(), 1, 0.0) == pytest.approx(0.5, abs=1e-6)
