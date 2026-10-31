"""Half the quoted spread in the names v27 traded (see configs/trading_costs_v28.yaml). A measurement, not a test. fetch samples SIP quotes
(read-only, resumable). report writes artifacts/reports/swing/v28_trading_costs.json.

    .venv/Scripts/python.exe scripts/v28_trading_costs.py fetch
    .venv/Scripts/python.exe scripts/v28_trading_costs.py report
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from itertools import islice
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

import numpy as np
import yaml

import v27_finrl_rotation as rotation
from atlas.feeds.alpaca import HISTORY_PATHS, HISTORY_QUOTES_PATH, http_fetch, iter_history, load_credentials
from atlas.swing.data import load_panel

cfg = yaml.safe_load(Path("configs/trading_costs_v28.yaml").read_text(encoding="utf-8"))
OUT, REPORT = Path(cfg["out"]), Path(cfg["report"])
NY = ZoneInfo("America/New_York")


def sample_days(dates: tuple[date, ...]) -> list[date]:
    """The first Monday session on or after the chosen day of each sampled month."""
    out = []
    for y in range(cfg["first_year"], cfg["last_day"].year + 1):
        for m in cfg["sample_months"]:
            day = next((d for d in dates if d.year == y and d.month == m and d.day >= cfg["first_day_of_month"] and d.weekday() == 0), None)
            if day is not None and day <= cfg["last_day"]:
                out.append(day)
    return out


def half_spread(quotes: list[dict]) -> float | None:
    """Median half-spread in bps over quotes with a real bid and ask."""
    bps = [(q["ap"] - q["bp"]) / (q["ap"] + q["bp"]) * 1e4 for q in quotes if q.get("bp", 0) > 0 and q.get("ap", 0) > q.get("bp", 0)]
    return median(bps) if bps else None


def names_by_year() -> dict[int, list[str]]:
    r = json.loads(rotation.OUT.read_text(encoding="utf-8"))
    finrl = [s for g in rotation.cfg["groups_v121"].values() for s in g]
    return {int(y): sorted({*finrl, *rotation.cfg["fallback"], *[s for g in groups.values() for s in g]}) for y, groups in r["free_lists"].items()}


def utc(day: date, hhmm: str) -> datetime:
    h, m, s = map(int, hhmm.split(":"))
    return datetime(day.year, day.month, day.day, h, m, s, tzinfo=NY).astimezone(timezone.utc)


def cmd_fetch() -> None:
    dates = load_panel(Path(rotation.cfg["bars_dir"]), [rotation.cfg["market"]]).dates
    fetch = http_fetch(load_credentials(Path(".env")), timeout_s=60, allowed=HISTORY_PATHS)
    names = names_by_year()
    done = set()
    if OUT.exists():
        done = {(r["day"], r["symbol"], r["window"]) for r in map(json.loads, OUT.read_text(encoding="utf-8").splitlines())}
    days = sample_days(dates)
    print("sessions", len(days), "names", max(map(len, names.values())), flush=True)
    with OUT.open("a", encoding="utf-8") as out:
        for i, day in enumerate(days):
            for symbol in names[day.year]:
                for window, at in cfg["windows"].items():
                    if (str(day), symbol, window) in done:
                        continue
                    start = utc(day, at)
                    stop = start + timedelta(seconds=cfg["window_seconds"])
                    fmt = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")
                    quotes = list(islice(iter_history(fetch, HISTORY_QUOTES_PATH, symbol, fmt(start), fmt(stop), "sip", limit=cfg["quotes_per_sample"], min_interval_s=cfg["min_interval_s"]), cfg["quotes_per_sample"]))
                    out.write(json.dumps(dict(day=str(day), symbol=symbol, window=window, half_spread=half_spread(quotes), quotes=len(quotes))) + "\n")
                    out.flush()
            print("day", i + 1, day, flush=True)
    print("quotes stored", flush=True)


def cmd_report() -> None:
    rows = [r for r in map(json.loads, OUT.read_text(encoding="utf-8").splitlines()) if r["half_spread"] is not None]
    etfs = {s for g in ("defensive",) for s in rotation.cfg["groups_v121"][g]} | set(rotation.cfg["fallback"]) | {"GLD", "SLV"}
    def line(pick) -> dict:
        v = np.array([r["half_spread"] for r in rows if pick(r)])
        return dict(samples=len(v), median=round(float(np.median(v)), 2), mean=round(float(v.mean()), 2), p90=round(float(np.percentile(v, 90)), 2))
    res: dict = {}
    for window in cfg["windows"]:
        res[window] = dict(stocks=line(lambda r: r["window"] == window and r["symbol"] not in etfs), etfs=line(lambda r: r["window"] == window and r["symbol"] in etfs),
                           stocks_2019_2021=line(lambda r: r["window"] == window and r["symbol"] not in etfs and r["day"] < "2022"), stocks_2022_on=line(lambda r: r["window"] == window and r["symbol"] not in etfs and r["day"] >= "2022"))
    REPORT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"fetch": cmd_fetch, "report": cmd_report}[sys.argv[1]]()
