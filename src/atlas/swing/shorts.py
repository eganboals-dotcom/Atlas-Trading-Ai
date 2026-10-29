"""Short interest from FINRA's twice-monthly files (``fetch-short``), point in time.

Each file is a settlement date's short positions, published about 8 business days later. A figure counts here
from ``PUBLISH_LAG_DAYS`` calendar days after its settlement date, safely after publication. ``short_features``
gives days-to-cover (short shares / average daily volume, as FINRA computes it) for every day and company;
NaN before a company's first published figure, and for everything before June 2021, when FINRA's files start
covering exchange-listed stocks.
"""

from __future__ import annotations

import io
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from atlas.swing.data import Panel

PUBLISH_LAG_DAYS = 14
MAX_AGE_DAYS = 45  # a figure older than about three settlement periods is stale
SCHEMA = {"symbol": pl.Utf8, "settlement": pl.Date, "short_qty": pl.Float64, "adv": pl.Float64, "dtc": pl.Float64}


def parse(data: bytes) -> pl.DataFrame:
    """One FINRA short-interest file (pipe-separated)."""
    df = pl.read_csv(io.BytesIO(data), separator="|", infer_schema=False, encoding="utf8-lossy",
                     truncate_ragged_lines=True)
    need = {"symbolCode", "settlementDate", "currentShortPositionQuantity", "averageDailyVolumeQuantity",
            "daysToCoverQuantity"}
    if not need <= set(df.columns):
        return pl.DataFrame(schema=SCHEMA)
    return df.select(
        pl.col("symbolCode").str.replace("/", ".").alias("symbol"),
        pl.col("settlementDate").str.to_date("%Y-%m-%d", strict=False).alias("settlement"),
        pl.col("currentShortPositionQuantity").cast(pl.Float64, strict=False).alias("short_qty"),
        pl.col("averageDailyVolumeQuantity").cast(pl.Float64, strict=False).alias("adv"),
        pl.col("daysToCoverQuantity").cast(pl.Float64, strict=False).alias("dtc")).drop_nulls(["symbol", "settlement"])


def short_features(p: Panel, table: pl.DataFrame) -> dict[str, np.ndarray]:
    """``short_dtc``: the latest days-to-cover already published before each day."""
    out = np.full(p.close.shape, np.nan)
    day_ord = np.array([d.toordinal() for d in p.dates])
    # FINRA writes share classes without the dot in some files (BRKB for BRK.B)
    names = {s: [s] + ([s.replace(".", "")] if "." in s else []) for s in p.symbols}
    wanted = [n for ns in names.values() for n in ns]
    by_sym = {key[0]: g.sort("settlement")
              for key, g in table.filter(pl.col("symbol").is_in(wanted)).group_by(["symbol"])}
    for j, sym in enumerate(p.symbols):
        g = next((by_sym[n] for n in names[sym] if n in by_sym), None)
        if g is None:
            continue
        avail = np.array([(d + timedelta(days=PUBLISH_LAG_DAYS)).toordinal() for d in g["settlement"].to_list()])
        settle = np.array([d.toordinal() for d in g["settlement"].to_list()])
        dtc = g["dtc"].to_numpy()
        k = np.searchsorted(avail, day_ord, side="left") - 1  # last figure available strictly before the day
        ok = k >= 0
        kk = np.clip(k, 0, None)
        fresh = ok & (day_ord - settle[kk] <= MAX_AGE_DAYS + PUBLISH_LAG_DAYS)
        out[fresh, j] = dtc[kk[fresh]]
    return {"short_dtc": out}


def available_from(settlement: date) -> date:
    return settlement + timedelta(days=PUBLISH_LAG_DAYS)


def load(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)
