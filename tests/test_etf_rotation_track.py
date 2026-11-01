"""The ETF rotation tracker: no ranking inside a group, later prices can't reach a book, and its rule stays frozen."""

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
import v32_etf_rotation_track as track  # noqa: E402

SIGNAL = date(2026, 10, 30)


def weekdays(after: int) -> list[date]:
    days, d = [], SIGNAL - timedelta(days=430)
    while len(days) < 1000:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days[: days.index(SIGNAL) + 1 + after]


def write_bars(cfg: dict, days: list[date], drift: dict[str, float] | None = None, shock: float = 1.0) -> None:
    names = track.names(cfg["rule"])
    rng = np.random.default_rng(5)
    r = rng.normal(0.0005, 0.01, (1000, len(names) + 1))
    for s, d in (drift or {}).items():
        r[:, names.index(s)] += d
    for j, s in enumerate([*names, "BIL"]):
        px = 100 * np.cumprod(1 + r[: len(days), j])
        px = np.where(np.array(days) > SIGNAL, px * shock, px)
        rows = pl.DataFrame(dict(date=days, open=px * 0.999, high=px, low=px, close=px, volume=np.full(len(days), 1e6)))
        folder = cfg["cash_dir"] if s == "BIL" else cfg["etf_dir"]
        Path(folder).mkdir(parents=True, exist_ok=True)
        rows.write_parquet(Path(folder) / f"{s}.parquet")


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
    c = yaml.safe_load((ROOT / "configs" / "etf_rotation_track_v32.yaml").read_text(encoding="utf-8"))
    c["etf_dir"], c["cash_dir"], c["out_dir"] = (str(tmp_path / x) for x in ("etf", "cash", "out"))
    return c


def at(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 9, 0, tzinfo=UTC)


def state(cfg: dict) -> dict:
    return json.loads((Path(cfg["out_dir"]) / "state.json").read_text())


def test_the_fallback_is_v27s_and_the_groups_are_only_etfs(cfg: dict) -> None:
    assert not set(track.names(cfg["rule"])) - {"XLK", "SMH", "XLE", "XLB", "GLD", "SLV", "TLT", "IEF", "XLU", "XLV", "IAU", "SHY", "UUP", "SPY", "QQQ"}
    assert set(rot.cfg["fallback"]) <= set(track.names(cfg["rule"]))
    assert cfg["rule"]["groups"]["defensive"] == rot.cfg["free_fixed"]["defensive"]


def test_a_kept_group_holds_every_member_equally(cfg: dict) -> None:
    write_bars(cfg, weekdays(0), drift=dict(XLE=0.004, XLB=0.004, GLD=0.004, SLV=0.004))
    m, _ = track.market(cfg, date(2026, 10, 31))
    w, b, kept = track.pick(m, len(m.ends) - 1, track.groups_of(m.p, cfg["rule"]))
    assert "real" in kept
    names = [m.p.symbols[j] for j in w]
    assert {"XLE", "XLB", "GLD", "SLV"} <= set(names)
    assert sum(w.values()) == pytest.approx(b)
    assert len({round(x, 9) for j, x in w.items() if m.p.symbols[j] in ("XLE", "XLB", "GLD", "SLV")}) == 1


def test_no_group_kept_means_the_fallback(cfg: dict) -> None:
    write_bars(cfg, weekdays(0), drift=dict(QQQ=0.02))
    m, _ = track.market(cfg, date(2026, 10, 31))
    w, b, kept = track.pick(m, len(m.ends) - 1, track.groups_of(m.p, cfg["rule"]))
    assert kept == ["fallback"] and sorted(w) == sorted(m.fallback)
    assert sum(w.values()) == pytest.approx(b)


def test_first_book_is_recorded_before_first_day(cfg: dict) -> None:
    write_bars(cfg, weekdays(0))
    assert "Not reviewed" in track.run(cfg, at(date(2026, 10, 31)))
    assert [b["signal"] for b in state(cfg)["books"]] == [str(SIGNAL)]


def test_later_prices_cannot_change_a_book(cfg: dict) -> None:
    write_bars(cfg, weekdays(0))
    track.run(cfg, at(date(2026, 10, 31)))
    first = state(cfg)["books"][0]
    write_bars(cfg, weekdays(15), shock=0.5)
    track.run(cfg, at(weekdays(15)[-1] + timedelta(days=1)))
    assert state(cfg)["books"][0] == first
    Path(cfg["out_dir"], "state.json").unlink()
    track.run(cfg, at(weekdays(15)[-1] + timedelta(days=1)))
    assert state(cfg)["books"][0]["weights"] == pytest.approx(first["weights"])


def test_scoring_starts_on_first_day(cfg: dict) -> None:
    days = weekdays(4)
    write_bars(cfg, days)
    track.run(cfg, at(days[-1] + timedelta(days=1)))
    m, cash = track.market(cfg, days[-1])
    r = track.score(cfg, m, cash, state(cfg)["books"])
    assert len(r["rotation"]) == 4 and len(r["plain"]) == 4
    assert r["SPY"][0] == pytest.approx(m.p.close[-4, m.spy] / m.p.close[-5, m.spy] - 1)


def test_changing_the_rule_is_refused(cfg: dict) -> None:
    write_bars(cfg, weekdays(0))
    track.run(cfg, at(date(2026, 10, 31)))
    cfg["rule"]["groups"]["growth"] = ["XLK"]
    with pytest.raises(SystemExit, match="changed"):
        track.run(cfg, at(date(2026, 10, 31)))
