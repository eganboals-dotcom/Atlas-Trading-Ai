"""v35: a 10-K's words, the similarity of two years' reports, and which two reports are compared."""

from __future__ import annotations

import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research" / "investing"))
import v35_lazy_prices as v  # noqa: E402


def test_words_skip_tags_hidden_xbrl_and_numbers() -> None:
    html = "<html><style>p{x:1}</style><ix:header><ix:hidden>dei:Foo 2024</ix:hidden></ix:header><p>Risk&nbsp;Factors: 12 NEW lawsuits, a</p></html>"
    assert v.words(html) == "risk factors new lawsuits"


def test_cosine_is_one_for_the_same_mix_and_lower_for_changes() -> None:
    a = Counter("risk risk sales".split())
    assert v.cosine(a, Counter("risk risk sales risk risk sales".split())) == pytest.approx(1.0)
    assert v.cosine(a, Counter("risk lawsuit lawsuit".split())) < v.cosine(a, Counter("risk risk sales lawsuit".split()))


def test_the_latest_report_is_paired_with_the_one_a_year_before() -> None:
    rows = [(date(2022, 2, 1), Path("a")), (date(2022, 9, 1), Path("amended-timing")), (date(2023, 2, 3), Path("b")), (date(2024, 2, 2), Path("c"))]
    assert v.pair(rows, date(2024, 2, 2)) == (Path("b"), Path("a"))  # filed that day isn't known yet
    assert v.pair(rows, date(2024, 2, 5)) == (Path("c"), Path("b"))
    assert v.pair(rows, date(2025, 6, 1)) is None  # the latest is too old
    assert v.pair(rows[:1], date(2022, 3, 1)) is None  # no report the year before
