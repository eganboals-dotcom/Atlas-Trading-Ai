"""The rotation against chance: a random book keeps its budget and weights, and draws only from what could be held."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v27_finrl_rotation as v  # noqa: E402
import v33_random_picks as r  # noqa: E402

GROUPS = {"defensive": [1, 2, 3, 4], "growth": [5, 6, 7, 8], "real": [9, 10, 11, 12]}


def book(slots: tuple) -> v.Book:
    weights = {j: w for members, ws in slots for j, w in zip(members, ws)}
    return v.Book(100, weights, slots, 0.8, ("growth",))


def test_a_random_book_keeps_its_weights_and_stays_inside_the_groups() -> None:
    k = book(((tuple(GROUPS["growth"]), (0.2, 0.2)), (tuple(GROUPS["real"]), (0.3, 0.1))))
    for seed in range(50):
        w = r.group_draw(k, GROUPS, np.random.default_rng(seed))
        assert sorted(w.values()) == pytest.approx([0.1, 0.2, 0.2, 0.3])
        held = {g for g, js in GROUPS.items() for j in w if j in js}
        assert len(held) == 2


def test_groups_are_really_drawn() -> None:
    k = book(((tuple(GROUPS["growth"]), (0.4, 0.4)),))
    held = {g for seed in range(50) for g, js in GROUPS.items() if set(r.group_draw(k, GROUPS, np.random.default_rng(seed))) & set(js)}
    assert held == set(GROUPS)


def test_a_fallback_book_stays_as_it_is() -> None:
    k = v.Book(100, {3: 0.3, 4: 0.3}, (), 0.6, ("fallback",))
    assert r.group_draw(k, GROUPS, np.random.default_rng(1)) == {3: 0.3, 4: 0.3}


def test_a_group_too_small_to_fill_the_book_cant_be_drawn() -> None:
    short = dict(GROUPS, real=[9])
    k = book(((tuple(GROUPS["growth"]), (0.2, 0.2, 0.2)),))
    for seed in range(50):
        assert not set(r.group_draw(k, short, np.random.default_rng(seed))) & {9}


def test_the_draws_are_repeatable() -> None:
    k = book(((tuple(GROUPS["growth"]), (0.2, 0.2)),))
    assert r.group_draw(k, GROUPS, np.random.default_rng(7)) == r.group_draw(k, GROUPS, np.random.default_rng(7))
