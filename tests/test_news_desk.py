"""News desk: what readers note, when the briefing knows it, and bots that can't see later news."""

from __future__ import annotations

import dataclasses
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from atlas.agents.genome import sample_genome
from atlas.news.desk import KeywordReader, briefing_features, render_briefing
from atlas.swing.data import Panel
from atlas.swing.families import FAMILIES, exposures
from atlas.timeutils import MARKET_TZ, datetime_to_ns


def articles(rows) -> pl.DataFrame:
    return pl.DataFrame([{"id": i, "created_ns": ns, "symbols": syms, "headline": h, "source": "test"}
                         for i, (ns, syms, h) in enumerate(rows)])


def at(d: date, hh: int, mm: int) -> int:
    return datetime_to_ns(datetime(d.year, d.month, d.day, hh, mm, tzinfo=MARKET_TZ))


def test_keyword_reader_notes_direction_importance_and_category() -> None:
    d = date(2024, 3, 7)
    notes = KeywordReader().read(articles([
        (at(d, 8, 0), ["IBM"], "Barclays Upgrades IBM to Overweight"),
        (at(d, 8, 1), ["MSFT"], "Microsoft Q4 EPS $1.46 Beats $1.37 Estimate"),
        (at(d, 8, 2), ["INTC"], "Intel Q2 EPS $(0.04) Misses $0.10 Estimate"),
        (at(d, 8, 3), ["JPM"], "JPMorgan earnings preview"),
        (at(d, 8, 4), ["AAPL", "MSFT", "IBM"], "Top Upgrades, Downgrades For March 7"),
        (at(d, 8, 5), ["AAPL"], "Apple unveils new iPhone"),
    ]))
    got = {r["symbol"]: (r["direction"], r["importance"], r["category"]) for r in notes.iter_rows(named=True)}
    assert got == {"IBM": (1.0, 0.6, "analyst"), "MSFT": (1.0, 1.0, "earnings"), "INTC": (-1.0, 1.0, "earnings"),
                   "JPM": (0.0, 0.3, "earnings")}  # the roundup and the product news are not noted


def make_panel(days: int = 600, n: int = 3) -> Panel:
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (days, n)), axis=0))
    opn = np.vstack([close[:1], close[:-1]])
    dates, d = [], date(2016, 1, 4)
    while len(dates) < days:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return Panel(tuple(dates), tuple(f"S{j}" for j in range(n)), opn, np.maximum(opn, close),
                 np.minimum(opn, close), close, np.full_like(close, 1e6))


def test_briefing_counts_a_note_only_from_the_first_open_after_it() -> None:
    p = make_panel()
    d0, d1 = p.dates[10], p.dates[11]
    notes = KeywordReader().read(articles([
        (at(d0, 8, 0), ["S0"], "S0 Q1 EPS $1.00 Beats $0.90 Estimate"),  # before the open of d0
        (at(d0, 11, 0), ["S1"], "Barclays Downgrades S1 to Underweight"),  # during d0: known at d1
        (at(d0, 17, 0), ["S2"], "Barclays Upgrades S2 to Overweight"),  # after the close: known at d1
    ]))
    f = briefing_features(p, notes)
    assert f["brief_score"][10].tolist() == [1.0, 0.0, 0.0]
    assert f["brief_score"][11].tolist() == [0.0, -0.6, 0.6]
    assert f["brief_count"].sum() == 3
    text = render_briefing(notes, p, 11)
    assert str(d1) in text and "| S1 | -0.60 | 1 |" in text


@pytest.mark.parametrize("family", ["brief_drift", "brief_tilt"])
def test_briefing_bots_see_nothing_after_the_open(family: str) -> None:
    p = make_panel()
    rng = np.random.default_rng(3)
    score = np.where(rng.random(p.close.shape) < 0.2, rng.normal(0, 1.5, p.close.shape), 0.0)
    p = dataclasses.replace(p, features={"brief_score": score})
    k = 400
    traded = False
    for _ in range(8):
        g = sample_genome(FAMILIES[family].space, rng)
        before = exposures(p, family, g)
        later = score.copy()
        later[k + 1:] = rng.normal(0, 3, later[k + 1:].shape)  # briefings after day k's open
        after = exposures(dataclasses.replace(p, features={"brief_score": later}), family, g)
        for b, a in zip(before, after):
            np.testing.assert_array_equal(b[:k + 1], a[:k + 1])
        traded |= before[0][:k].sum() > 0
    assert traded


def test_brief_tilt_stays_fully_invested_and_drops_bad_news() -> None:
    p = make_panel()
    score = np.zeros(p.close.shape)
    score[300, 0] = -2.0  # bad news on S0 before the open of day 300
    p = dataclasses.replace(p, features={"brief_score": score})
    xi, _ = exposures(p, "brief_tilt", {"window": 5, "thr": 1.0, "tilt": 0.5, "rebalance": 1})
    assert np.allclose(xi[1:].sum(axis=1), 3)  # always all capital (3 slots)
    assert xi[300, 0] == 0 and xi[304, 0] == 0 and xi[305, 0] > 0  # out for the 5-open window


def test_ai_reader_hides_names_maps_notes_back_and_ignores_invented_ones() -> None:
    from atlas.news.llm_reader import LlmReader, anonymize

    assert anonymize("Johnson & Johnson Q3 EPS $2.10 Beats; J&J Raises Outlook", ["JNJ"]) == \
        "Company A Q3 EPS $2.10 Beats; Company A Raises Outlook"
    assert anonymize("Citi Cuts Price Target on Visa to $300", ["C", "V"]) == \
        "Company A Cuts Price Target on Company B to $300"
    assert anonymize("T-Mobile and AT&T sign deal", ["T"]) == "T-Mobile and Company A sign deal"

    sent = []

    def fake(items):
        sent.extend(items)
        out = []
        for it in items:
            for c in it["companies"]:
                out.append({"i": it["i"], "company": c, "direction": 2.0, "importance": 0.5, "category": "deal"})
        out.append({"i": 99, "company": "Company A", "direction": 1, "importance": 1, "category": "other"})  # invented
        out.append({"i": 0, "company": "Company C", "direction": 1, "importance": 1, "category": "other"})  # invented
        return out

    d = date(2024, 3, 7)
    arts = articles([(at(d, 7 + i // 60, i % 60), syms, h) for i, (syms, h) in enumerate([
        (["AAPL"], "Apple signs deal"), (["AAPL", "MSFT"], "Apple and Microsoft partner"),
        (["AAPL", "MSFT", "IBM"], "Tech roundup"), (["XOM"], "Exxon outlook"),
        (["IBM"], "Xylem stock plunges after cutting its outlook"), (["MSFT", "IBM"], "Microsoft cuts prices")] * 15)])
    notes = LlmReader(fake, batch_size=7, tracked={"AAPL", "MSFT", "IBM"}).read(arts)
    assert all("Apple" not in it["headline"] and "Microsoft" not in it["headline"] for it in sent)
    # roundups (3 companies), untracked companies and companies the headline never names are never sent
    assert len(sent) == 45 and all(it["companies"] == ["Company A"] for it in sent[2::3])  # IBM isn't named
    assert notes.height == 60 and set(notes["symbol"]) == {"AAPL", "MSFT"}
    assert notes["direction"].max() == 1.0 and set(notes["reader"]) == {"claude"}  # clipped to [-1, 1]
    # two share classes of one company: asked about once, as one company, and noted for both tickers
    sent.clear()
    g = LlmReader(fake, tracked={"GOOGL", "MSFT"}).read(articles([(at(d, 9, 0), ["GOOG", "GOOGL", "MSFT"],
                                                                   "Google and Microsoft sign cloud deal")]))
    assert sent[0]["companies"] == ["Company A", "Company B"] and "Google" not in sent[0]["headline"]
    assert sorted(g["symbol"]) == ["GOOG", "GOOGL", "MSFT"]


def test_daily_desk_fetches_only_new_headlines_and_never_rereads(tmp_path) -> None:
    from datetime import UTC

    from atlas.news.daily import DeskConfig, briefing_window, next_open, run_desk

    cfg = DeskConfig(start_date=date(2026, 10, 29), symbols=["AAPL", "MSFT"], news_dir=tmp_path / "news",
                     notes_dir=tmp_path / "notes", out_dir=tmp_path / "out", readers=["keyword", "local", "claude"])
    d = date(2026, 10, 29)
    feed = [{"id": 1, "created_at": "2026-10-29T15:00:00Z", "symbols": ["AAPL"],
             "headline": "Barclays Upgrades Apple to Overweight"},
            {"id": 2, "created_at": "2026-10-29T13:30:00Z", "symbols": ["MSFT"],
             "headline": "Microsoft Q1 EPS $3.10 Beats $2.90 Estimate"}]
    asked, sent = [], []

    def fetch(syms, start, end):
        asked.append((start, end))
        return [a for a in feed if start <= a["created_at"] < end]

    def factory():
        def classify(items):
            sent.extend(items)
            return [{"i": it["i"], "company": "Company A", "direction": 0.5, "importance": 1.0, "category": "other"}
                    for it in items]
        return classify

    now = datetime(2026, 10, 29, 18, 0, tzinfo=MARKET_TZ).astimezone(UTC)  # after the close
    logs: list[str] = []
    out = run_desk(cfg, fetch, {"local": factory, "claude": "no key"}, now=now, log=logs.append)
    assert asked[0][0] == "2026-10-29T04:00:00Z" and len(sent) == 2
    assert "claude reader skipped: no key" in logs
    text = out.read_text(encoding="utf-8")
    assert "open of 2026-10-30" in text and "| MSFT | +1.00 | 1 |" in text and "| AAPL | +0.50 | 1 |" in text
    assert "## Reader: local" in text and "## Reader: claude\n\nNo notes" in text
    # second run: asks only for headlines after the last one stored, and sends nothing new to the AI reader
    feed.append({"id": 3, "created_at": "2026-10-29T20:30:00Z", "symbols": ["AAPL"], "headline": "Apple news"})
    run_desk(cfg, fetch, {"local": factory}, now=now + timedelta(hours=1), log=lambda m: None)
    assert asked[1][0] == "2026-10-29T15:00:01Z" and len(sent) == 3
    # a reader that couldn't run (Ollama off) catches up on everything it missed the next time it can
    run_desk(cfg, None, {"local": factory, "claude": factory}, now=now + timedelta(hours=2), log=lambda m: None)
    assert len(sent) == 6
    assert next_open(datetime(2026, 10, 30, 17, 0, tzinfo=MARKET_TZ)) == date(2026, 11, 2)  # Friday -> Monday
    lo, hi = briefing_window(date(2026, 11, 2))
    assert lo == at(date(2026, 10, 30), 9, 15) and hi == at(d + timedelta(days=4), 9, 15)
