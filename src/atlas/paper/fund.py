"""Paper fund: equal capital per bot, compounding day to day.

* Day 1: ``total_capital`` is split equally across the bots (the champion and shadows;
  the benchmark is not a bot). Each bot trades only its own share.
* Every day after: each bot starts with exactly what it ended the previous trading day
  with. Its capital grows and shrinks with its own P&L, and money never moves between
  bots.
* At the close, open positions are sold at the bid, after costs (``flatten_at_close``).
  A bot's day therefore ends in cash, and nobody gets a free exit at the mid price.
* The buy-and-hold benchmark gets ``total_capital / bots`` fresh every day. It is a
  yardstick for returns, not part of the fund.

The ledger is a JSON file written atomically. A day's entry is overwritten if the same
day is settled again (e.g. after a stop and resume), so settling is idempotent.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


class FundError(RuntimeError):
    pass


class FundLedger:
    def __init__(self, path: Path, total_capital: float) -> None:
        self.path = path
        self.total_capital = total_capital
        self.data: dict[str, Any] | None = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    @property
    def exists(self) -> bool:
        return self.data is not None

    def start_capitals(self, bots: dict[str, str], day: date) -> tuple[dict[str, float], dict[str, float], float]:
        """(capital at the start of ``day``, capital at inception) per bot, and the benchmark's capital.

        ``bots`` maps account id -> agent id. A different lineup than the ledger's is refused,
        so P&L is never attributed to the wrong bot.
        """
        share = self.total_capital / len(bots)
        if self.data is None:
            self.data = {
                "created": datetime.now(UTC).isoformat(), "total_capital": self.total_capital,
                "bots": {b: {"agent_id": a, "inception_capital": share} for b, a in sorted(bots.items())},
                "days": [],
            }
            return {b: share for b in bots}, {b: share for b in bots}, share
        known = {b: v["agent_id"] for b, v in self.data["bots"].items()}
        if known != bots:
            changed = sorted(set(known.items()) ^ set(bots.items()))
            raise FundError(f"bot lineup differs from the fund's ({changed}); keep the lineup, or start a new fund "
                            "with --reset-fund (the old ledger is archived, not deleted)")
        prior = [d for d in self.data["days"] if d["date"] < day.isoformat()]
        inception = {b: v["inception_capital"] for b, v in self.data["bots"].items()}
        start = {b: (prior[-1]["bots"][b]["end"] if prior else inception[b]) for b in bots}
        return start, inception, self.data["total_capital"] / len(bots)

    def settle(self, day: date, results: dict[str, dict[str, float]], final: bool, note: str) -> None:
        assert self.data is not None
        entry = {"date": day.isoformat(), "final": final, "note": note, "settled_at": datetime.now(UTC).isoformat(),
                 "bots": {b: {**r, "pnl": r["end"] - r["start"]} for b, r in sorted(results.items())},
                 "fund_end": sum(r["end"] for r in results.values())}
        self.data["days"] = [d for d in self.data["days"] if d["date"] != entry["date"]] + [entry]
        self.data["days"].sort(key=lambda d: d["date"])
        self.save()

    def save(self) -> None:
        assert self.data is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def archive(self) -> Path | None:
        """Move the current ledger aside (``--reset-fund``); nothing is deleted."""
        if not self.path.is_file():
            return None
        target = self.path.with_name(f"{self.path.stem}.{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json")
        os.replace(self.path, target)
        self.data = None
        return target

    def days_traded(self) -> int:
        return len(self.data["days"]) if self.data else 0
