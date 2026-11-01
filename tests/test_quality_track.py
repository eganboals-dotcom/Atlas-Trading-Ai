"""The earnings-quality tracker: its rule stays frozen, books use only past filings and never change, and a gone company stays in."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "trackers"))
import v24_quality_track as track  # noqa: E402

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
FIRST = date(2026, 11, 2)


def weekdays(n: int) -> list[date]:
    """n sessions from the Friday before first_day on."""
    days, d = [], date(2026, 10, 30)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def write_bars(cfg: dict, days: list[date], gone: dict[str, date] | None = None) -> None:
    bars = Path(cfg["bars_dir"])
    bars.mkdir(parents=True, exist_ok=True)
    for j, s in enumerate(["BIL", "SPY", *SYMBOLS]):
        kept = [d for d in days if d < (gone or {}).get(s, date.max)]
        px = [100.0 * (1 + 0.001 * j) ** i for i in range(len(kept))]
        pl.DataFrame(dict(date=kept, open=px, close=px), schema={"date": pl.Date, "open": pl.Float64, "close": pl.Float64}).write_parquet(bars / f"{s}.parquet")


def filing(sym: str, cfg: dict, income: float, cash: float, filed: date = date(2026, 2, 10), year: int = 2025) -> None:
    rows = []
    for y, scale in ((year, 1.0), (year - 1, 0.9)):
        end = date(y, 12, 31)
        rows += [(k, end, v * scale, filed) for k, v in (("assets", 1000.0), ("net_income", income), ("cfo", cash), ("gross_profit", 300.0))]
    path = Path(cfg["filings_dir"]) / f"{sym}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pl.DataFrame(rows, schema=track.SCHEMA, orient="row")
    (pl.concat([pl.read_parquet(path), new]) if path.exists() else new).write_parquet(path)


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    c = yaml.safe_load((ROOT / "configs" / "quality_track_v24.yaml").read_text(encoding="utf-8"))
    c["bars_dir"], c["filings_dir"], c["out_dir"] = (str(tmp_path / x) for x in ("bars", "sec", "out"))
    c["spy_dir"] = c["bars_dir"]
    c["rule"]["min_eligible"] = 2
    c["rule"]["every"] = 3
    monkeypatch.setattr(track, "universe", lambda rule: list(SYMBOLS))
    monkeypatch.setattr(track, "refresh", lambda cfg, symbols: None)
    for s, cash in zip(SYMBOLS, (500.0, 400.0, 300.0, 200.0, 100.0)):
        filing(s, c, 100.0 if s != "AAA" else -50.0, cash)
    return c


def state(cfg: dict) -> dict:
    return json.loads((Path(cfg["out_dir"]) / "state.json").read_text())


def test_the_rule_matches_what_v23_tested() -> None:
    rule = yaml.safe_load((ROOT / "configs" / "quality_track_v24.yaml").read_text(encoding="utf-8"))["rule"]
    v23 = yaml.safe_load((ROOT / "configs" / "filings_v23.yaml").read_text(encoding="utf-8"))
    assert (rule["universe_config"], rule["every"], rule["top_share"], rule["min_eligible"], rule["cost_bps"]) == (
        v23["swing_config"], v23["every"], v23["top_share"], v23["min_eligible"], v23["cost_bps"])


def test_first_book_is_chosen_before_first_day(cfg: dict) -> None:
    write_bars(cfg, weekdays(1))
    track.run(cfg, date(2026, 10, 31))
    b = state(cfg)["books"]
    assert [x["cut"] for x in b] == ["2026-10-30"] and b[0]["top"] == ["AAA"] and b[0]["bottom"] == ["EEE"]
    assert b[0]["profitable"] == ["BBB"]  # AAA made a loss
    assert "Not reviewed" in (Path(cfg["out_dir"]) / "tracker.md").read_text(encoding="utf-8")


def test_a_filing_after_the_last_close_changes_nothing(cfg: dict) -> None:
    write_bars(cfg, weekdays(1))
    filing("EEE", cfg, -500.0, 900.0, filed=date(2026, 10, 31), year=2026)
    track.run(cfg, date(2026, 10, 31))
    assert state(cfg)["books"][0]["top"] == ["AAA"]


def test_next_book_comes_only_after_each_books_last_session(cfg: dict) -> None:
    days = weekdays(8)
    write_bars(cfg, days[:3])  # book 1 has had two of its three sessions
    track.run(cfg, days[3])
    assert len(state(cfg)["books"]) == 1
    write_bars(cfg, days[:4])
    track.run(cfg, days[4])
    assert [x["cut"] for x in state(cfg)["books"]] == [str(days[0]), str(days[3])]


def test_a_recorded_book_never_changes(cfg: dict) -> None:
    write_bars(cfg, weekdays(2))
    track.run(cfg, date(2026, 11, 3))
    first = state(cfg)["books"]
    for s in SYMBOLS:
        (Path(cfg["filings_dir"]) / f"{s}.parquet").unlink()
    track.run(cfg, date(2026, 11, 3))
    assert state(cfg)["books"] == first


def test_book_earns_only_from_first_day(cfg: dict) -> None:
    days = weekdays(4)
    write_bars(cfg, days)
    track.run(cfg, days[-1] + timedelta(days=1))
    m = track.market(cfg, SYMBOLS)
    r = track.score(cfg, m, state(cfg)["books"])
    assert len(r["top"]) == 3 and m.p.dates[1] == FIRST
    c = cfg["rule"]["cost_bps"] / 1e4
    assert r["top"][1] == pytest.approx(m.p.close[2, 0] / m.p.close[1, 0] - 1)
    assert r["top"][2] == pytest.approx(m.p.close[3, 0] / m.p.close[2, 0] - 1 - c)  # sold at the last close


def test_scoring_stops_where_the_chosen_books_end(cfg: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    days = weekdays(10)
    write_bars(cfg, days[:1])
    track.run(cfg, date(2026, 10, 31))
    write_bars(cfg, days)

    def down(cfg: dict, symbols: list[str]) -> None:
        raise OSError("EDGAR down")

    monkeypatch.setattr(track, "refresh", down)
    text = track.run(cfg, days[-1])
    assert len(state(cfg)["books"]) == 1 and "waits for SEC filings" in text
    assert len(track.score(cfg, track.market(cfg, SYMBOLS), state(cfg)["books"])["top"]) == cfg["rule"]["every"]


def test_a_gone_company_stays_in_and_the_calendar_goes_on(cfg: dict) -> None:
    days = weekdays(7)
    write_bars(cfg, days, gone=dict(AAA=days[2]))
    text = track.run(cfg, days[-1] + timedelta(days=1))
    m = track.market(cfg, SYMBOLS)
    assert m.p.dates == tuple(days)
    assert m.p.close[-1, 0] == m.p.close[1, 0] and m.p.open[-1, 0] == m.p.close[1, 0]
    assert "Stopped trading, held at the last close: AAA" in text
    b = state(cfg)["books"]
    assert b[0]["top"] == ["AAA"] and b[1]["top"] == ["BBB"] and b[1]["eligible"] == 4  # it can't be bought again


def test_changing_the_rule_is_refused(cfg: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    write_bars(cfg, weekdays(1))
    track.run(cfg, date(2026, 10, 31))
    cfg["rule"]["top_share"] = 0.1
    with pytest.raises(SystemExit, match="changed"):
        track.run(cfg, date(2026, 10, 31))
    cfg["rule"]["top_share"] = 0.2
    monkeypatch.setattr(track, "universe", lambda rule: SYMBOLS[:-1])
    with pytest.raises(SystemExit, match="changed"):
        track.run(cfg, date(2026, 10, 31))


def test_no_verdict_before_the_review() -> None:
    rule = dict(review_sessions=40, bootstrap_samples=200, block=5, seed=1)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.002, 40)
    hold, rf = 0.0005 + noise, np.zeros(40)
    good = dict(top=hold + 0.002 + rng.normal(0, 0.0002, 40), Hold=hold, bottom=hold)
    assert "Not reviewed" in track.verdict(rule, {k: v[:39] for k, v in good.items()}, rf)
    assert "PASS" in track.verdict(rule, good, rf)
    assert "FAIL" in track.verdict(rule, good | dict(top=hold - 0.002), rf)
    assert "FAIL" in track.verdict(rule, good | dict(bottom=hold + 0.01), rf)  # must also beat the control
    assert "Not reviewed" in track.verdict(rule, {}, rf[:0])
