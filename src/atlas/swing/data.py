"""Daily bars: download (read-only Alpaca history), store as local Parquet, align into a panel."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl

from atlas.timeutils import MARKET_TZ, NS_PER_S

DEFAULT_DIR = Path("data/processed/alpaca_daily")
COLUMNS = ("open", "high", "low", "close", "volume")


def daily_rows(bars: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Alpaca 1Day bars -> rows keyed by the market date (bar timestamps are midnight ET)."""
    from atlas.feeds.alpaca import parse_rfc3339_ns

    rows = []
    for b in bars:
        d = datetime.fromtimestamp(parse_rfc3339_ns(b["t"]) / NS_PER_S, UTC).astimezone(MARKET_TZ).date()
        rows.append({"date": d, "open": float(b["o"]), "high": float(b["h"]), "low": float(b["l"]),
                     "close": float(b["c"]), "volume": float(b.get("v") or 0.0)})
    return rows


def save_daily(rows_by_symbol: dict[str, list[dict[str, Any]]], out_dir: Path, meta: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for sym, rows in rows_by_symbol.items():
        pl.DataFrame(rows).sort("date").write_parquet(out_dir / f"{sym}.parquet")
    manifest = {**meta, "symbols": {s: {"days": len(r), "first": str(r[0]["date"]), "last": str(r[-1]["date"])}
                                    for s, r in rows_by_symbol.items() if r}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class Panel:
    """Aligned daily bars, shape (days, symbols). Only dates every symbol traded are kept."""

    dates: tuple[date, ...]
    symbols: tuple[str, ...]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    features: dict[str, np.ndarray] = field(default_factory=dict)  # extra (days, symbols) data, e.g. news counts

    def before(self, end: date) -> Panel:
        """The panel cut to dates strictly before ``end`` (e.g. to keep a holdout out of reach)."""
        n = sum(d < end for d in self.dates)
        return Panel(self.dates[:n], self.symbols, *(getattr(self, c)[:n] for c in COLUMNS),
                     {k: v[:n] for k, v in self.features.items()})

    def select(self, symbols: Iterable[str]) -> Panel:
        idx = [self.symbols.index(s) for s in symbols]
        return Panel(self.dates, tuple(self.symbols[i] for i in idx), *(getattr(self, c)[:, idx] for c in COLUMNS),
                     {k: v[:, idx] for k, v in self.features.items()})

    @property
    def overnight(self) -> np.ndarray:
        """Return from the close of t-1 to the open of t (0 on the first day)."""
        r = np.zeros_like(self.close)
        r[1:] = self.open[1:] / self.close[:-1] - 1.0
        return r

    @property
    def intraday(self) -> np.ndarray:
        """Return from the open of t to the close of t."""
        return self.close / self.open - 1.0


def load_panel(data_dir: Path = DEFAULT_DIR, symbols: Iterable[str] | None = None) -> Panel:
    syms = list(symbols) if symbols is not None else sorted(p.stem for p in Path(data_dir).glob("*.parquet"))
    frames = {s: pl.read_parquet(Path(data_dir) / f"{s}.parquet") for s in syms}
    common = sorted(set.intersection(*(set(f["date"].to_list()) for f in frames.values())))
    arrays = {c: np.empty((len(common), len(syms))) for c in COLUMNS}
    for j, s in enumerate(syms):
        f = frames[s].filter(pl.col("date").is_in(common)).sort("date")
        for c in COLUMNS:
            arrays[c][:, j] = f[c].to_numpy()
    return Panel(tuple(common), tuple(syms), *(arrays[c] for c in COLUMNS))
