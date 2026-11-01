"""One page across every forward tracker: each rule against Hold and SPY, and anything that needs a person. It reads their reports and SPY's bars,
and changes nothing, so no frozen rule is touched. SPY's column is a yardstick here and no part of any verdict. Writes
artifacts/reports/summary/latest.md. Research only: no order is placed anywhere.

    .venv/Scripts/python.exe scripts/tracker_summary.py
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import polars as pl

REPORTS = Path("artifacts/reports")
SPY = Path("data/processed/swing_track/SPY.parquet")
LOG = REPORTS / "swing_track" / "runs.log"
SWING = ("swing_track", "swing_track_news", "swing_track_fund", "swing_track_u300", "swing_track_signals", "swing_track_v12")
TABLES = {"quality_track": "Earnings quality", "rotation_track": "Rotation", "etf_rotation_track": "ETF rotation"}
LONG = 60  # the trackers themselves show annual figures only after this many days
OUT = REPORTS / "summary" / "latest.md"


def spy_return(spy: pl.DataFrame, first: date, last: date) -> float | None:
    """SPY from the close before first to the close of last, dividends included."""
    before = spy.filter(pl.col("date") < first)["close"]
    end = spy.filter(pl.col("date") <= last)["close"]
    return float(end[-1] / before[-1] - 1) if len(before) and len(end) else None


def swing_rows(folder: Path) -> list[dict]:
    if not (folder / "tracker.json").exists():
        return []
    rows = []
    for uni, u in json.loads((folder / "tracker.json").read_text(encoding="utf-8"))["universes"].items():
        hold = u["rows"]["Hold"].get("total_return")
        for name, r in u["rows"].items():
            if name != "Hold" and "total_return" in r:
                rows.append(dict(tracker=f"{folder.name.removeprefix('swing_track').strip('_') or 'swing'}/{uni}", rule=name, days=r["days"], first=date.fromisoformat(u["first_day"]),
                                 last=date.fromisoformat(u["last_day"]), ret=r["total_return"], hold=hold))
    return rows


def table_row(text: str, label: str) -> dict | None:
    """The first rule of a tracker.md with a Days line and a table, as v24, v30 and v32 write them."""
    days = re.search(r"Days (\d{4}-\d\d-\d\d) to (\d{4}-\d\d-\d\d)", text)
    row = re.search(r"^\| " + re.escape(label) + r"[^|]*\| ([+-][\d.]+)%", text, re.M)
    hold = re.search(r"^\| Hold \| ([+-][\d.]+)%", text, re.M)
    if not days or not row:
        return None
    return dict(first=date.fromisoformat(days[1]), last=date.fromisoformat(days[2]), ret=float(row[1]) / 100, hold=float(hold[1]) / 100 if hold else None)


def table_rows() -> list[dict]:
    rows = []
    for folder, label in TABLES.items():
        path = REPORTS / folder / "tracker.md"
        row = table_row(path.read_text(encoding="utf-8"), label) if path.exists() else None
        if row:
            rows.append(dict(tracker=folder.removesuffix("_track"), rule=label, days=None, **row))
    return rows


def run_problems(text: str) -> list[str]:
    """What the last nightly run in the log says went wrong."""
    last = text[text.rfind("=== "):] if "=== " in text else text
    found = [f"Last run: {line.strip()[:150]}" for line in last.splitlines() if re.search(r"Traceback|^refused|Error:|\bfailed\b", line)]
    return found[:5]


def report_notes() -> list[str]:
    """Lines the trackers flag for a person."""
    out = []
    for folder in (*TABLES, "fomc_track"):
        path = REPORTS / folder / "tracker.md"
        if path.exists():
            out += [f"{folder}: {line.removeprefix('- ')}" for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("- ") and not line.startswith("- Next decision")]
    return out


def verdicts() -> list[str]:
    out = []
    for folder in (*TABLES, "fomc_track"):
        path = REPORTS / folder / "tracker.md"
        if path.exists():
            found = re.search(r"\*\*(.+?)\*\*", path.read_text(encoding="utf-8"))
            out.append(f"{folder.removesuffix('_track')}: {found[1] if found else 'no verdict line'}")
    return out


def stale(rows: list[dict], newest: date) -> list[str]:
    return sorted({f"{r['tracker']} stops at {r['last']}, SPY's bars at {newest}" for r in rows if r["last"] < newest})


def percent(x: float | None) -> str:
    return "" if x is None else f"{x:+.2%}"


def build(rows: list[dict], spy: pl.DataFrame, problems: list[str], notes: list[str], status: list[str], today: date) -> str:
    newest = spy["date"].max()
    lines = ["# Atlas trackers on one page", "", f"Written {today}, prices through {newest}. Research only: no order is placed anywhere, and nothing here is advice.", ""]
    flags = [*problems, *stale(rows, newest), *notes]
    lines += ["## Needs a look", ""] + ([f"- {x}" for x in flags] or ["Nothing."]) + [""]
    short = min((r["days"] for r in rows if r["days"] is not None), default=0)
    lines += ["## Verdicts", ""] + [f"- {x}" for x in status] + ["", "## Each rule", ""]
    if short < LONG:
        lines += [f"Only {short} days so far. Days this few say almost nothing about any rule.", ""]
    lines += ["| Tracker | Rule | Return | vs Hold | vs SPY | From |", "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["tracker"], r["rule"])):
        s = spy_return(spy, r["first"], r["last"])
        lines.append(f"| {r['tracker']} | {r['rule']} | {percent(r['ret'])} | {percent(None if r['hold'] is None else r['ret'] - r['hold'])} | {percent(None if s is None else r['ret'] - s)} | {r['first']} |")
    lines += ["", "SPY runs from the close before each rule's first day, with dividends. A rule can beat Hold and still lose to SPY, because Hold here is a list of survivors.", ""]
    return "\n".join(lines)


def main() -> None:
    spy = pl.read_parquet(SPY).sort("date")
    rows = [r for f in SWING for r in swing_rows(REPORTS / f)] + table_rows()
    text = build(rows, spy, run_problems(LOG.read_text(encoding="utf-8", errors="replace")) if LOG.exists() else [], report_notes(), verdicts(), date.today())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
