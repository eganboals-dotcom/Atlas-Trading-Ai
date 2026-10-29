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
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "RevenuesNetOfInterestExpense"],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
                        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],  # Mastercard 2014 on, IBM before 2015: ProfitLoss only
}
# Some companies (J&J, P&G, AT&T, Verizon) report equity only including minority interests.
INSTANTS = {"assets": ["Assets"],
            "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]}
# Companies with several share classes (Alphabet, Meta) report no single cover-page count; the balance-sheet
# count or the diluted average count covers all classes.
SHARES = [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding"),
          ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding")]
# Health insurers file medical claims outside cost of revenue, so revenue minus cost isn't a gross profit
# (UnitedHealth would show 89%); like banks, they get none and are skipped by gpa rankings.
NO_GROSS_PROFIT = frozenset({"UNH", "CVS"})
MAX_AGE_DAYS = 550  # a figure is used until 18 months after its period ends (a missed annual report, a stopped
                    # series, Berkshire's Class A share counts that end in 2011), then it counts as unknown
SCHEMA = {"item": pl.Utf8, "end": pl.Date, "val": pl.Float64, "filed": pl.Date}
FEATURES = ("fund_gpa", "fund_roa", "fund_bm", "fund_ep")


def _rows(facts: dict[str, Any], concepts: list[tuple[str, str]], unit: str, annual: bool, instant: bool,
          only_outside: bool = False) -> list[tuple[date, float, date]]:
    """Companies switch concepts over the years (Meta: Revenues until 2017, then
    RevenueFromContractWithCustomerExcludingAssessedTax; Thermo Fisher: CostOfGoodsSold until 2017), so every
    concept's rows are kept, and where two concepts give the same period in the same filing the earlier-listed
    one wins. ``only_outside``: a later concept only adds periods before or after those already covered (share
    counts, where a balance-sheet count must not interleave with cover-page counts)."""
    kept: list[tuple[date, float, date]] = []
    seen: set[tuple[date, date]] = set()
    for taxonomy, concept in concepts:
        entries = facts.get("facts", {}).get(taxonomy, {}).get(concept, {}).get("units", {}).get(unit, [])
        lo, hi = (min(r[0] for r in kept), max(r[0] for r in kept)) if kept else (None, None)
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
            if only_outside and lo is not None and lo <= end <= hi:
                continue
            if (end, filed) not in seen:
                out.append((end, float(e["val"]), filed))
        seen |= {(r[0], r[2]) for r in out}
        kept += out
    return kept


def extract(facts: dict[str, Any]) -> pl.DataFrame:
    rows = []
    for item, concepts in ANNUAL_FLOWS.items():
        rows += [(item, *r) for r in _rows(facts, [("us-gaap", c) for c in concepts], "USD", annual=True,
                                           instant=False)]
    for item, concepts in INSTANTS.items():
        rows += [(item, *r) for r in _rows(facts, [("us-gaap", c) for c in concepts], "USD", annual=True,
                                           instant=True)]
    rows += [("shares", *r) for r in _rows(facts, SHARES, "shares", annual=False, instant=True, only_outside=True)]
    return pl.DataFrame(rows, schema=SCHEMA, orient="row") if rows else pl.DataFrame(schema=SCHEMA)


def as_of(rows: pl.DataFrame, item: str, dates: tuple[date, ...], max_age_days: int = MAX_AGE_DAYS) -> np.ndarray:
    """For each day, the value of ``item`` for the latest period among reports filed before that day, while
    that period ended at most ``max_age_days`` earlier."""
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
        if best_end is not None and (d - best_end).days <= max_age_days:
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
    gp[:, [j for j, s in enumerate(p.symbols) if s in NO_GROSS_PROFIT]] = np.nan
    prev_raw = np.vstack([np.full((1, shape[1]), np.nan), raw_close[:-1]])
    mcap = prev_raw * get["shares"]
    with np.errstate(invalid="ignore", divide="ignore"):
        assets = np.where(get["assets"] > 0, get["assets"], np.nan)
        mcap = np.where(mcap > 0, mcap, np.nan)
        return {"fund_gpa": gp / assets, "fund_roa": get["net_income"] / assets, "fund_bm": get["equity"] / mcap,
                "fund_ep": get["net_income"] / mcap}
