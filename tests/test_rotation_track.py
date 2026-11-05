"""The rotation tracker: its rule stays frozen, books use only past closes and never change, and sleeves change one at a time."""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "trackers"))
import v27_finrl_rotation as rot  # noqa: E402
import v30_rotation_track as track  # noqa: E402

SYMBOLS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
GROUPS = {"growth": SYMBOLS[:4], "real": SYMBOLS[4:]}
SIGNAL = date(2026, 10, 30)


def weekdays(after: int) -> list[date]:
    """About 300 sessions through first_signal, then after more."""
    days, d = [], SIGNAL - timedelta(days=430)
    while len(days) < 1000:
        if d.weekday() < 5 and d != date(2026, 11, 26):
            days.append(d)
        d += timedelta(days=1)
    return days[: days.index(SIGNAL) + 1 + after]


def write_bars(cfg: dict, days: list[date], gone: dict[str, date] | None = None, shock: float = 1.0) -> None:
    """Prices from a fixed draw, so a day's price doesn't depend on how many days exist. shock scales everything after first_signal."""
    names = ["BIL", *SYMBOLS, *track.all_etfs()]
    rng = np.random.default_rng(5)
    r = rng.normal(0.0005, 0.01, (1000, len(names)))
    for j, s in enumerate(names):
        px = 100 * np.cumprod(1 + r[: len(days), j])
        px = np.where(np.array(days) > SIGNAL, px * shock, px)
        rows = pl.DataFrame(dict(date=days, open=px * 0.999, close=px, volume=np.full(len(days), 1e6)), schema={"date": pl.Date, "open": pl.Float64, "close": pl.Float64, "volume": pl.Float64})
        rows = rows.filter(pl.col("date") < (gone or {}).get(s, date.max))
        folder = cfg["bars_dir"] if s == "BIL" or s in SYMBOLS else cfg["etf_dir"]
        Path(folder).mkdir(parents=True, exist_ok=True)
        rows.write_parquet(Path(folder) / f"{s}.parquet")


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    c = yaml.safe_load((ROOT / "configs" / "rotation_track_v30.yaml").read_text(encoding="utf-8"))
    c["bars_dir"], c["etf_dir"], c["out_dir"] = (str(tmp_path / x) for x in ("bars", "etf", "out"))
    monkeypatch.setattr(track, "universe", lambda rule: list(SYMBOLS))
    monkeypatch.setattr(track, "refresh", lambda cfg, now: None)
    monkeypatch.setattr(track, "year_list", lambda mk, year: GROUPS)
    return c


def state(cfg: dict) -> dict:
    return json.loads((Path(cfg["out_dir"]) / "state.json").read_text())


def at(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 9, 0, tzinfo=UTC)


def scored(cfg: dict, day: date) -> tuple[track.Market, dict]:
    mk = track.market(cfg, SYMBOLS, day)
    return mk, track.score(cfg, mk, state(cfg))


def test_the_rule_matches_what_v27_and_v29_tested() -> None:
    rule = yaml.safe_load((ROOT / "configs" / "rotation_track_v30.yaml").read_text(encoding="utf-8"))["rule"]
    assert (rule["universe_config"], rule["cost_bps"], rule["cash_symbol"]) == (rot.cfg["swing_config"], rot.cfg["cost_bps"], rot.cfg["cash_symbol"])
    assert date.fromisoformat(rule["first_day"]) == date(2026, 11, 2)
    assert date.fromisoformat(rule["first_signal"]).weekday() == 4


def test_a_friday_counts_only_once_it_is_over() -> None:
    days = tuple(weekdays(0)[-3:])
    assert track.closed_weeks(days, SIGNAL + timedelta(days=1))[-1] == 2
    assert 2 not in track.closed_weeks(days, SIGNAL)


def test_sleeves_change_one_at_a_time() -> None:
    books = [{"X": 1.0}, {"Y": 1.0}, {"Z": 1.0}, {"W": 1.0}, {"V": 1.0}, {"U": 1.0}]
    mix = track.sleeve_mix(books, 4)
    assert mix[0] == {"X": 1.0}
    assert mix[1] == {"X": 0.75, "Y": 0.25}
    assert mix[3] == {"X": 0.25, "Y": 0.25, "Z": 0.25, "W": 0.25}
    assert mix[4] == {"V": 0.25, "Y": 0.25, "Z": 0.25, "W": 0.25}  # the sleeve the oldest book sat in
    assert mix[5] == {"V": 0.25, "U": 0.25, "Z": 0.25, "W": 0.25}


def test_first_book_is_recorded_before_first_day(cfg: dict) -> None:
    write_bars(cfg, weekdays(0))
    text = track.run(cfg, at(date(2026, 10, 31)))
    b = state(cfg)["books"]
    assert [x["signal"] for x in b] == [str(SIGNAL)] and b[0]["weights"]
    assert "Not reviewed" in text


def test_later_prices_cannot_change_a_book(cfg: dict) -> None:
    write_bars(cfg, weekdays(0))
    track.run(cfg, at(date(2026, 10, 31)))
    first = state(cfg)["books"]
    write_bars(cfg, weekdays(15), shock=0.5)
    track.run(cfg, at(weekdays(15)[-1] + timedelta(days=1)))
    assert state(cfg)["books"][0] == first[0]
    Path(cfg["out_dir"], "state.json").unlink()  # chosen afresh, knowing the later prices, it comes out the same
    track.run(cfg, at(weekdays(15)[-1] + timedelta(days=1)))
    assert state(cfg)["books"][0]["weights"] == pytest.approx(first[0]["weights"])


def test_book_earns_only_from_first_day_at_the_open(cfg: dict) -> None:
    days = weekdays(4)
    write_bars(cfg, days)
    track.run(cfg, at(days[-1] + timedelta(days=1)))
    mk, r = scored(cfg, days[-1])
    assert len(r["rotation"]) == 4
    assert r["SPY"][0] == pytest.approx(mk.m.p.close[-4, mk.m.spy] / mk.m.p.close[-5, mk.m.spy] - 1)
    bought = dict(state(cfg)["books"][0]["weights"])
    p = mk.m.p
    idx = {s: j for j, s in enumerate(p.symbols)}
    t = p.dates.index(date(2026, 11, 2))
    open_to_close = sum(w * (p.close[t, idx[s]] / p.open[t, idx[s]] - 1) for s, w in bought.items())
    assert r["rotation"][0] < open_to_close + mk.cash[t] + 1e-9  # paid costs, and earned nothing from before the open


def test_scoring_stops_where_the_books_end(cfg: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    write_bars(cfg, weekdays(0))
    track.run(cfg, at(date(2026, 10, 31)))
    days = weekdays(12)
    write_bars(cfg, days)

    def down(cfg: dict, now: datetime) -> None:
        raise OSError("Alpaca down")

    monkeypatch.setattr(track, "refresh", down)
    text = track.run(cfg, at(days[-1] + timedelta(days=1)))
    assert len(state(cfg)["books"]) == 1 and "waits for ETF bars" in text
    assert len(scored(cfg, days[-1])[1]["rotation"]) == 5  # through the next Friday's close, the last session the first book can hold


def test_a_gone_company_cant_be_picked_and_the_calendar_goes_on(cfg: dict) -> None:
    days = weekdays(0)
    write_bars(cfg, days, gone=dict(A1=days[-3], A2=days[-3], A3=days[-3], B1=days[-3], B2=days[-3], B3=days[-3]))
    cfg["rule"]["min_stock_bars"] = 0.2
    track.run(cfg, at(date(2026, 10, 31)))
    mk = track.market(cfg, SYMBOLS, date(2026, 10, 31))
    assert mk.m.p.dates[-1] == SIGNAL
    assert not {"A1", "A2", "A3", "B1", "B2", "B3"} & set(state(cfg)["books"][0]["weights"])


def test_a_session_missing_most_stocks_isnt_scored(cfg: dict) -> None:
    days = weekdays(1)
    write_bars(cfg, days, gone={s: days[-1] for s in SYMBOLS[:4]})
    assert track.market(cfg, SYMBOLS, days[-1]).m.p.dates[-1] == SIGNAL


def test_changing_the_rule_is_refused(cfg: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    write_bars(cfg, weekdays(0))
    track.run(cfg, at(date(2026, 10, 31)))
    cfg["rule"]["cost_bps"] = 2.0
    with pytest.raises(SystemExit, match="changed"):
        track.run(cfg, at(date(2026, 10, 31)))
    cfg["rule"]["cost_bps"] = 10.0
    monkeypatch.setitem(rot.rule, "top_assets", 2)
    with pytest.raises(SystemExit, match="changed"):
        track.run(cfg, at(date(2026, 10, 31)))


def test_no_verdict_before_the_review() -> None:
    rule = dict(review_sessions=40, bootstrap_samples=200, block=5, seed=1)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.002, 40)
    spy, rf = 0.0005 + noise, np.zeros(40)
    good = dict(rotation=spy + 0.002 + rng.normal(0, 0.0002, 40), SPY=spy, plain=spy, rf=rf)
    assert "Not reviewed" in track.verdict(rule, {k: x[:39] for k, x in good.items()})
    assert "PASS" in track.verdict(rule, good)
    assert "FAIL" in track.verdict(rule, good | dict(rotation=spy - 0.002))
    assert "FAIL" in track.verdict(rule, good | dict(plain=spy + 0.01))  # must also beat the list held equally
    assert "Not reviewed" in track.verdict(rule, {})
