"""v38: last week's return uses only real bars before the rebalance, and turnover counts the holding replaced."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research" / "investing"))
import v38_reversal as v  # noqa: E402


def test_score_is_last_weeks_return_from_real_bars_only() -> None:
    close = np.array([[10.0, 5.0, 8.0]] * 6 + [[11.0, 4.0, 8.0], [99.0, 99.0, 99.0]])  # row 7 is the rebalance day, never read
    traded = np.ones_like(close, dtype=bool)
    traded[1, 2] = False  # the third company's close 6 sessions back is a filled-in price
    s = v.scores(close, traded, 7)
    assert s[:2] == pytest.approx([0.1, -0.2])
    assert np.isnan(s[2])


def test_turnover_is_the_share_replaced() -> None:
    assert v.turnover([[1, 2, 3, 4], [1, 2, 5, 6], [1, 2, 5, 6]]) == pytest.approx(0.25)
