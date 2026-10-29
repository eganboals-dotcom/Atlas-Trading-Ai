"""Forward tracker: rules can't change after freezing, only completed days after the start count."""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from atlas.swing.data import Panel, save_daily
from atlas.swing.track import TrackConfig, check_frozen, last_complete_day, score, track
from atlas.timeutils import MARKET_TZ


def make_cfg(tmp_path, **over) -> TrackConfig:
    d = {"start_date": date(2017, 6, 1), "history_start": date(2016, 1, 4), "out_dir": tmp_path / "out",
         "dataset_dir": tmp_path / "daily", "cash_symbol": "CASH", "cost_bps": 0.0,
         "universes": {"u": ["S0", "S1", "S2"]},
         "rules": [{"name": "mom", "universe": "u", "family": "rotation",
                    "genome": {"mode": 0.0, "lookback": 60, "skip": 0, "top_k": 1, "rebalance": 21, "abs_filter": 0.0}}]}
    d.update(over)
    return TrackConfig.model_validate(d)


def write_data(cfg: TrackConfig, days: int = 600, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    syms = ["S0", "S1", "S2", "CASH"]
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (days, 4)), axis=0))
    close[:, 3] = 100 * 1.0001 ** np.arange(days)
    opn = np.vstack([close[:1], close[:-1] * np.exp(rng.normal(0, 0.004, (days - 1, 4)))])
    dates = []
    d = date(2016, 1, 4)
    while len(dates) < days:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    rows = {s: [{"date": dates[i], "open": opn[i, j], "high": max(opn[i, j], close[i, j]),
                 "low": min(opn[i, j], close[i, j]), "close": close[i, j], "volume": 1e6} for i in range(days)]
            for j, s in enumerate(syms)}
    save_daily(rows, cfg.dataset_dir, {"source": "test"})
    return Panel(tuple(dates), tuple(syms), opn, np.maximum(opn, close), np.minimum(opn, close), close,
                 np.full_like(close, 1e6))


def test_rules_cannot_change_after_the_first_run(tmp_path) -> None:
    cfg = make_cfg(tmp_path)
    first = check_frozen(cfg)
    assert check_frozen(cfg)["fingerprint"] == first["fingerprint"]
    assert check_frozen(make_cfg(tmp_path, dataset_dir=tmp_path / "elsewhere"))  # where data lives doesn't matter
    tweaked = make_cfg(tmp_path)
    rule = tweaked.rules[0].model_dump()
    rule["genome"]["lookback"] = 61
    for changed in (make_cfg(tmp_path, rules=[rule]), make_cfg(tmp_path, start_date=date(2017, 7, 1)),
                    make_cfg(tmp_path, cost_bps=1.0)):
        with pytest.raises(ValueError, match="changed"):
            check_frozen(changed)


def test_todays_bar_counts_only_once_it_is_final() -> None:
    def at(h, m):
        return datetime(2026, 10, 29, h, m, tzinfo=MARKET_TZ).astimezone(UTC)

    assert last_complete_day(at(10, 0)) == date(2026, 10, 28)
    assert last_complete_day(at(16, 5)) == date(2026, 10, 28)  # closing prints can still arrive
    assert last_complete_day(at(16, 30)) == date(2026, 10, 29)


def test_only_days_from_the_start_count(tmp_path) -> None:
    cfg = make_cfg(tmp_path)
    p = write_data(cfg)
    res = score(cfg, {"u": p})["universes"]["u"]
    s = next(i for i, d in enumerate(p.dates) if d >= cfg.start_date)
    hold = res["rows"]["Hold"]
    assert res["first_day"] == str(p.dates[s]) and hold["days"] == len(p.dates) - s
    want = np.mean(p.close[-1, :3] / p.open[s, :3]) - 1  # not exact: Hold rebalances daily to equal weight
    assert hold["total_return"] == pytest.approx(want, abs=0.05)
    single = make_cfg(tmp_path, universes={"u": ["S1"]}, rules=[])
    one = score(single, {"u": p})["universes"]["u"]["rows"]["Hold"]
    first_day = p.close[s, 1] / p.open[s, 1] - 1 + 0.5 * 0.0001  # cash idle overnight before the first open
    assert math.isclose(one["total_return"], (1 + first_day) * p.close[-1, 1] / p.close[s, 1] - 1, rel_tol=1e-9)
    # prices before the start move the signals but not the scored returns of Hold
    q = Panel(p.dates, p.symbols, p.open.copy(), p.high.copy(), p.low.copy(), p.close.copy(), p.volume)
    q.close[: s - 1] *= 1.3
    q.open[: s - 1] *= 1.3
    assert score(single, {"u": q})["universes"]["u"]["rows"]["Hold"]["total_return"] == pytest.approx(
        one["total_return"], rel=1e-12)


def test_tracker_writes_a_report_and_handles_no_days_yet(tmp_path) -> None:
    cfg = make_cfg(tmp_path)
    write_data(cfg)
    out = track(cfg)
    text = out.read_text(encoding="utf-8")
    assert "| Hold |" in text and "| mom |" in text and "no order is placed" in text
    rep = json.loads((cfg.out_dir / "tracker.json").read_text(encoding="utf-8"))
    assert rep["fingerprint"] == json.loads((cfg.out_dir / "frozen.json").read_text(encoding="utf-8"))["fingerprint"]
    later = make_cfg(tmp_path / "b", start_date=date(2030, 1, 1), dataset_dir=cfg.dataset_dir)
    assert "No completed days" in track(later).read_text(encoding="utf-8")


def test_fund_rank_rules_hold_the_most_profitable_from_filed_figures(tmp_path) -> None:
    import polars as pl

    from atlas.swing.fundamentals import SCHEMA

    rule = {"name": "quality", "universe": "u", "family": "fund_rank",
            "genome": {"factor": 0.0, "top_k": 1, "rebalance": 63}}
    cfg = make_cfg(tmp_path, rules=[rule], fundamentals_dir=tmp_path / "sec", raw_dataset_dir=tmp_path / "daily")
    write_data(cfg)
    cfg.fundamentals_dir.mkdir()
    for sym, gp in (("S0", 10.0), ("S1", 50.0), ("S2", 30.0)):  # S1 the most profitable, from its 2016 10-K on
        rows = [("gross_profit", date(2016, 12, 31), gp, date(2017, 2, 20)),
                ("assets", date(2016, 12, 31), 100.0, date(2017, 2, 20))]
        pl.DataFrame(rows, schema=SCHEMA, orient="row").write_parquet(cfg.fundamentals_dir / f"{sym}.parquet")
    out = track(cfg).read_text(encoding="utf-8")
    assert "| quality |" in out
    rep = json.loads((cfg.out_dir / "tracker.json").read_text(encoding="utf-8"))
    assert rep["universes"]["u"]["rows"]["quality"]["holding_last_day"] == {"S1": 1.0}
    with pytest.raises(ValueError, match="fundamentals_dir"):  # a fund_rank rule without filings is refused
        track(make_cfg(tmp_path / "b", rules=[rule], dataset_dir=cfg.dataset_dir, raw_dataset_dir=cfg.dataset_dir))
    # where the filings live doesn't change the fingerprint, and old trackers keep theirs
    assert check_frozen(make_cfg(tmp_path, rules=[rule], fundamentals_dir=tmp_path / "x", raw_dataset_dir=tmp_path))


@pytest.mark.parametrize("path", ["configs/swing_track.yaml", "configs/swing_track_news.yaml",
                                  "configs/swing_track_fund.yaml", "configs/swing_track_u300.yaml",
                                  "configs/swing_track_signals.yaml"])
def test_the_shipped_tracker_configs_are_valid(path: str) -> None:
    from pathlib import Path

    from atlas.config import load_config
    from atlas.swing.families import FAMILIES

    cfg = load_config(Path(path), TrackConfig)
    assert cfg.start_date == date(2026, 10, 29)
    for r in cfg.rules:
        assert r.universe in cfg.universes and set(r.genome) == set(FAMILIES[r.family].space)
