"""Per-run performance metrics.

Definitions (Phase 2; Phase 3 adds Sharpe/Sortino, profit factor, and cost scenarios):

* ``net_return`` — final equity / initial cash - 1, after all modeled spread, slippage,
  fees, and fill effects.
* ``max_drawdown`` — largest peak-to-trough equity decline, as a fraction of the peak,
  on the engine's ~1 s equity samples plus every fill.
* ``downside`` — root-mean-square of negative per-sample equity returns, scaled by
  sqrt(number of samples) to a per-window figure.
* ``turnover`` — total traded notional / initial cash.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from atlas.environment.orders import OrderStatus
from atlas.environment.replay import RunResult


@dataclass(frozen=True, slots=True)
class RunMetrics:
    net_return: float
    max_drawdown: float
    downside: float
    turnover: float
    fills: int
    orders: int
    rejected: int
    fees: float
    final_equity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def max_drawdown(equity: list[float]) -> float:
    peak = -math.inf
    worst = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, (peak - v) / peak)
    return worst


def downside_deviation(equity: list[float]) -> float:
    rets = [b / a - 1.0 for a, b in zip(equity, equity[1:]) if a > 0]
    if not rets:
        return 0.0
    neg_sq = [min(r, 0.0) ** 2 for r in rets]
    return math.sqrt(sum(neg_sq) / len(rets)) * math.sqrt(len(rets))


def run_metrics(result: RunResult, initial_cash: float) -> RunMetrics:
    curve = [e for _, e in result.equity_curve]
    # Equity right after each fill sharpens the drawdown estimate between samples.
    fill_equity = [r["portfolio_equity"] for r in result.audit if r.get("fill_qty")]
    final = result.final["equity"]
    series = [initial_cash, *curve, final] if curve else [initial_cash, final]
    return RunMetrics(
        net_return=final / initial_cash - 1.0,
        max_drawdown=max_drawdown(sorted_merge(result, series, fill_equity)),
        downside=downside_deviation(series),
        turnover=sum(f.quantity * f.price for f in result.fills) / initial_cash,
        fills=len(result.fills),
        orders=len(result.orders),
        rejected=sum(o.status is OrderStatus.REJECTED for o in result.orders),
        fees=result.final["fees_paid"],
        final_equity=final,
    )


def sorted_merge(result: RunResult, series: list[float], fill_equity: list[float]) -> list[float]:
    """Time-ordered union of curve samples and post-fill equity (for drawdown only)."""
    if not fill_equity:
        return series
    points = [(ts, e) for ts, e in result.equity_curve]
    points += [(r["ts"], r["portfolio_equity"]) for r in result.audit if r.get("fill_qty")]
    points.sort(key=lambda p: p[0])
    return [series[0], *(e for _, e in points), series[-1]]
