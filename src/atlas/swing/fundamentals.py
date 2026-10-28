"""Company fundamentals from SEC filings, point in time: a value counts from the day after it was filed.

``extract`` keeps the few annual figures the value and quality factors need from a company's EDGAR
``companyfacts`` (10-K values for fiscal years; shares outstanding from any filing). ``fundamental_features``
turns them into (days, symbols) arrays for a panel: on day t, each company's latest fiscal year among the
reports filed before t. Market value uses the unadjusted close of t-1 times the latest reported share count,
because filed share counts are not split-adjusted.

Factors (higher = more attractive in each):
* ``fund_gpa`` gross profitability, gross profit / assets (Novy-Marx, 2013); companies without gross profit
  (banks, insurers) have none and are skipped by gpa rankings
* ``fund_roa`` net income / assets
* ``fund_bm`` book-to-market, equity / market value (Fama and French)
* ``fund_ep`` earnings yield, net income / market value
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from atlas.swing.data import Panel

ANNUAL_FLOWS = {  # name -> us-gaap concepts in order of preference
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "net_income": ["NetIncomeLoss"],
}
INSTANTS = {"assets": ["Assets"], "equity": ["StockholdersEquity"]}
SCHEMA = {"item": pl.Utf8, "end": pl.Date, "val": pl.Float64, "filed": pl.Date}
FEATURES = ("fund_gpa", "fund_roa", "fund_bm", "fund_ep")


def _rows(facts: dict[str, Any], taxonomy: str, concepts: list[str], unit: str, annual: bool, instant: bool
          ) -> list[tuple[date, float, date]]:
    for concept in concepts:  # the first concept the company actually reports
        entries = facts.get("facts", {}).get(taxonomy, {}).get(concept, {}).get("units", {}).get(unit, [])
        out = []
        for e in entries:
            if annual and not (e.get("form", "").startswith("10-K") and e.get("fp") == "FY"):
                continue
            end, filed = date.fromisoformat(e["end"]), date.fromisoformat(e["filed"])
            if annual and not instant:
                if "start" not in e:
                    continue
                days = (end - date.fromisoformat(e["start"])).days
                if not 330 <= days <= 400:  # a full fiscal year, not a quarter in the annual report
                    continue
            out.append((end, float(e["val"]), filed))
        if out:
            return out
    return []


def extract(facts: dict[str, Any]) -> pl.DataFrame:
    rows = []
    for item, concepts in ANNUAL_FLOWS.items():
        rows += [(item, *r) for r in _rows(facts, "us-gaap", concepts, "USD", annual=True, instant=False)]
    for item, concepts in INSTANTS.items():
        rows += [(item, *r) for r in _rows(facts, "us-gaap", concepts, "USD", annual=True, instant=True)]
    rows += [("shares", *r) for r in _rows(facts, "dei", ["EntityCommonStockSharesOutstanding"], "shares",
                                           annual=False, instant=True)]
    return pl.DataFrame(rows, schema=SCHEMA, orient="row") if rows else pl.DataFrame(schema=SCHEMA)


def as_of(rows: pl.DataFrame, item: str, dates: tuple[date, ...]) -> np.ndarray:
    """For each day, the value of ``item`` for the latest period among reports filed before that day."""
    r = rows.filter(pl.col("item") == item).sort(["filed", "end"])
    out = np.full(len(dates), np.nan)
    if r.height == 0:
        return out
    best_end, best_val, k = None, np.nan, 0
    items = r.select("end", "val", "filed").rows()
    for i, d in enumerate(dates):
        while k < len(items) and items[k][2] < d:  # filed before day d: public by its open
            end, val, _ = items[k]
            if best_end is None or end >= best_end:  # a later period, or a later filing of the same period
                best_end, best_val = end, val
            k += 1
        out[i] = best_val
    return out


def fundamental_features(p: Panel, fundamentals_dir: Path, raw_close: np.ndarray) -> dict[str, np.ndarray]:
    """``raw_close``: unadjusted closes aligned with ``p`` (for market value)."""
    shape = p.close.shape
    get = {k: np.full(shape, np.nan) for k in ("revenue", "gross_profit", "cost_of_revenue", "net_income", "assets",
                                                "equity", "shares")}
    for j, sym in enumerate(p.symbols):
        f = Path(fundamentals_dir) / f"{sym}.parquet"
        if f.exists():
            rows = pl.read_parquet(f)
            for k in get:
                get[k][:, j] = as_of(rows, k, p.dates)
    gp = np.where(np.isnan(get["gross_profit"]), get["revenue"] - get["cost_of_revenue"], get["gross_profit"])
    prev_raw = np.vstack([np.full((1, shape[1]), np.nan), raw_close[:-1]])
    mcap = prev_raw * get["shares"]
    with np.errstate(invalid="ignore", divide="ignore"):
        assets = np.where(get["assets"] > 0, get["assets"], np.nan)
        mcap = np.where(mcap > 0, mcap, np.nan)
        return {"fund_gpa": gp / assets, "fund_roa": get["net_income"] / assets, "fund_bm": get["equity"] / mcap,
                "fund_ep": get["net_income"] / mcap}
