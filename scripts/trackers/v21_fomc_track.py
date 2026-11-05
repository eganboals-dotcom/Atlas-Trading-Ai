"""Forward tracker for the pre-FOMC drift (see configs/fomc_track_v21.yaml, which holds the frozen rule and its verdict).

Downloads SPY's 14:00 bars for finished sessions (read-only Alpaca market data), scores new Fed decisions and writes
artifacts/reports/fomc_track/tracker.md. Research only: no order is placed anywhere.

    .venv/Scripts/python.exe scripts/trackers/v21_fomc_track.py
    .venv/Scripts/python.exe scripts/trackers/v21_fomc_track.py check   # compares a fetch with the stored bars, and saves nothing
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import yaml

from atlas.feeds.alpaca import HISTORY_BARS_PATH, HISTORY_PATHS, http_fetch, iter_history, load_credentials

CONFIG = Path("configs/fomc_track_v21.yaml")
ET = ZoneInfo("America/New_York")


def fetch_prices(cfg: dict, first: date, last: date) -> pl.DataFrame:
    """day and the open of the decision minute, one row per session that has it."""
    if first > last:
        return pl.DataFrame(schema={"day": pl.Date, "o": pl.Float64})
    fetch = http_fetch(load_credentials(Path(".env")), timeout_s=60, allowed=HISTORY_PATHS)
    minute = cfg["rule"]["decision_minute"]
    rows = []
    for b in iter_history(fetch, HISTORY_BARS_PATH, cfg["rule"]["symbol"], f"{first}T00:00:00Z", f"{last + timedelta(days=1)}T04:00:00Z",
                          cfg["feed"], min_interval_s=cfg["min_interval_s"], extra={"timeframe": "1Min", "adjustment": "split"}):
        et = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET)
        if et.hour * 60 + et.minute - 570 == minute and first <= et.date() <= last:
            rows.append(dict(day=et.date(), o=float(b["o"])))
    return pl.DataFrame(rows, schema={"day": pl.Date, "o": pl.Float64})


def update_prices(cfg: dict, today: date) -> pl.DataFrame:
    """Stored prices plus every finished session since. Today is left out, so a part-day is never scored."""
    path = Path(cfg["bars"])
    old = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema={"day": pl.Date, "o": pl.Float64})
    first = old["day"].max() + timedelta(days=1) if old.height else date.fromisoformat(cfg["rule"]["first_day"])
    new = fetch_prices(cfg, first, today - timedelta(days=1))
    out = pl.concat([old, new]).sort("day")
    if new.height:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(path)
    return out


def score(rule: dict, decisions: set[date], prices: pl.DataFrame) -> tuple[dict[date, float], list[float]]:
    """Return of each event, and of every other window."""
    days, px = prices["day"].to_list(), prices["o"].to_numpy()
    c = rule["cost_bps"] / 1e4
    events, other = {}, []
    for i in range(1, len(days)):
        r = float(px[i] * (1 - c) / (px[i - 1] * (1 + c)) - 1)
        if days[i] in decisions:
            events[days[i]] = r
        elif days[i - 1] not in decisions:
            other.append(r)
    return events, other


def fingerprint(rule: dict) -> str:
    return hashlib.sha256(json.dumps(rule, sort_keys=True).encode("utf-8")).hexdigest()


def check_state(cfg: dict, decisions: set[date], state_path: Path) -> dict:
    """Refuses a changed rule, or a list that has dropped an event already scored."""
    fp = fingerprint(cfg["rule"])
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else dict(fingerprint=fp, frozen=str(date.today()), scored={})
    if state["fingerprint"] != fp:
        sys.exit("refused: the frozen rule in configs/fomc_track_v21.yaml has changed")
    gone = [d for d in state["scored"] if date.fromisoformat(d) not in decisions]
    if gone:
        sys.exit(f"refused: scored events were removed from the decisions list: {gone}")
    return state


def verdict(rule: dict, ev: np.ndarray, other: list[float]) -> str:
    n = rule["review_events"]
    if len(ev) < n:
        return f"Not reviewed yet: {len(ev)} of {n} events. Nothing before then is a verdict."
    rng = np.random.default_rng(rule["seed"])
    boot = ev[rng.integers(0, len(ev), (rule["bootstrap_samples"], len(ev)))].mean(axis=1)
    lo = float(np.percentile(boot, 2.5))
    won = ev.mean() > 0 and lo > 0 and ev.mean() > np.mean(other)
    return f"Review after {len(ev)} events: {'PASS' if won else 'FAIL'} (interval's low end {lo * 1e4:+.1f} bps)."


def report(cfg: dict, state: dict, other: list[float], decisions: set[date], today: date) -> str:
    rule = cfg["rule"]
    ev = np.array(list(state["scored"].values()))
    upcoming = sorted(d for d in decisions if d >= today)
    lines = ["# Pre-FOMC drift: forward tracker", "", f"Frozen {state['frozen']}. Research only: no order is placed anywhere.", "",
             f"**{verdict(rule, ev, other)}**", ""]
    if len(ev):
        lines += [f"- Events: {len(ev)}, mean {ev.mean() * 1e4:+.1f} bps, up on {(ev > 0).mean():.0%}", ""]
    if other:
        lines += [f"- Other windows: {len(other)}, mean {np.mean(other) * 1e4:+.1f} bps", ""]
    lines += [f"- Next decision: {upcoming[0] if upcoming else 'none listed: add the new year to the config'}", ""]
    if len(ev):
        lines += ["| Decision | Return |", "|---|---|"] + [f"| {d} | {r * 1e4:+.1f} bps |" for d, r in sorted(state["scored"].items())] + [""]
    return "\n".join(lines)


def run(cfg: dict, today: date) -> str:
    decisions = {d for year in cfg["decisions"] for d in year}
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = check_state(cfg, decisions, state_path)
    events, other = score(cfg["rule"], decisions, update_prices(cfg, today))
    for d, r in events.items():
        state["scored"].setdefault(str(d), r)  # a score never changes once recorded
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    text = report(cfg, state, other, decisions, today)
    (out / "tracker.md").write_text(text, encoding="utf-8")
    return text


def cmd_check(cfg: dict) -> None:
    """A week before the holdout, fetched as the tracker fetches it, against the bars v15 stored."""
    first, last = date(2026, 4, 20), date(2026, 4, 24)
    got = fetch_prices(cfg, first, last)
    old = (pl.read_parquet(f"data/processed/etf_1min_ohlcv/{cfg['rule']['symbol']}.parquet")
           .filter((pl.col("m") == cfg["rule"]["decision_minute"]) & pl.col("day").is_between(first, last)).select("day", "o"))
    if got.height != 5 or not got.sort("day").equals(old.sort("day")):
        sys.exit(f"check failed: {got} vs {old}")
    print("check ok:", got.height, "sessions match the stored bars")


if __name__ == "__main__":
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if sys.argv[1:] == ["check"]:
        cmd_check(config)
    else:
        print(run(config, datetime.now(ET).date()))
