"""Per-run performance metrics, with every definition written down.

* ``net_return`` — final equity / initial cash - 1. Net of all modeled spread, slippage,
  fees, and fill effects.
* ``max_drawdown`` — largest peak-to-trough fall in equity, as a fraction of the peak.
  Measured on the engine's ~1 s equity samples plus the equity right after every fill.
* ``downside`` — root-mean-square of negative per-sample equity returns, scaled by
  sqrt(number of samples) to a per-window figure.
* ``turnover`` — total traded notional / initial cash.
* ``sharpe`` / ``sortino`` — computed from 1-minute equity returns within the session,
  annualized by sqrt(252 * 390), with a zero risk-free rate. Sortino uses the RMS of
  negative minute returns. Both are ``None`` with fewer than 2 returns or zero
  dispersion. Short sessions give few samples, so treat them as noisy.
* Round trips — a round trip runs from flat to flat in one symbol. Its P&L is net of
  every fee on its fills.
  * ``profit_factor`` = gross profit / gross loss over round trips. ``None`` when there
    are no round trips, or no losing ones (see ``losing_trades``).
  * ``win_rate`` = winners / round trips.
  * ``mean_trade_pnl`` = mean round-trip P&L in currency.
* ``fill_rate`` — filled quantity / submitted quantity over every proposed order,
  rejected ones included.
* ``avg_exposure`` / ``peak_exposure`` — market value / equity, averaged and maximized
  over the equity samples.
* ``symbol_pnl`` — per symbol, net cash flow from fills minus fees, plus the value of
  any open position at its last mark.

Fills and returns at this frequency are serially correlated, so treat per-trade
statistics as descriptive, not as independent observations.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from atlas.environment.orders import OrderStatus, Side
from atlas.environment.replay import RunResult
from atlas.timeutils import NS_PER_S

MINUTES_PER_YEAR = 252 * 390


@dataclass(frozen=True)
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
    sharpe: float | None = None
    sortino: float | None = None
    round_trips: int = 0
    losing_trades: int = 0
    profit_factor: float | None = None
    win_rate: float | None = None
    mean_trade_pnl: float | None = None
    fill_rate: float | None = None
    avg_exposure: float = 0.0
    peak_exposure: float = 0.0
    symbol_pnl: dict[str, float] = field(default_factory=dict)

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


def minute_returns(curve: list[tuple[int, float]], initial: float) -> list[float]:
    if not curve:
        return []
    start = curve[0][0]
    points = [initial]
    boundary = start + 60 * NS_PER_S
    last = initial
    for ts, eq in curve:
        while ts >= boundary:
            points.append(last)
            boundary += 60 * NS_PER_S
        last = eq
    points.append(curve[-1][1])
    return [b / a - 1.0 for a, b in zip(points, points[1:]) if a > 0]


def sharpe_sortino(rets: list[float]) -> tuple[float | None, float | None]:
    if len(rets) < 2:
        return None, None
    mean = statistics.fmean(rets)
    sd = statistics.pstdev(rets)
    down = math.sqrt(sum(min(r, 0.0) ** 2 for r in rets) / len(rets))
    scale = math.sqrt(MINUTES_PER_YEAR)
    return (mean / sd * scale if sd > 0 else None), (mean / down * scale if down > 0 else None)


def round_trip_pnls(result: RunResult) -> list[float]:
    qty: dict[str, int] = defaultdict(int)
    open_pnl: dict[str, float] = defaultdict(float)
    trips: list[float] = []
    for f in result.fills:
        sign = -1.0 if f.side is Side.BUY else 1.0
        open_pnl[f.symbol] += sign * f.quantity * f.price - f.fee
        qty[f.symbol] += f.quantity if f.side is Side.BUY else -f.quantity
        if qty[f.symbol] == 0:
            trips.append(open_pnl.pop(f.symbol))
    return trips


def symbol_pnl(result: RunResult) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for f in result.fills:
        out[f.symbol] += (-1.0 if f.side is Side.BUY else 1.0) * f.quantity * f.price - f.fee
    marks = result.final.get("marks", {})
    for sym, pos in result.final["positions"].items():
        if pos["quantity"]:
            out[sym] += pos["quantity"] * marks.get(sym, pos["avg_cost"])
    return dict(sorted(out.items()))


def run_metrics(result: RunResult, initial_cash: float) -> RunMetrics:
    curve = [e for _, e in result.equity_curve]
    final = result.final["equity"]
    series = [initial_cash, *curve, final] if curve else [initial_cash, final]
    sharpe, sortino = sharpe_sortino(minute_returns(result.equity_curve, initial_cash))

    trips = round_trip_pnls(result)
    gains = sum(p for p in trips if p > 0)
    losses = -sum(p for p in trips if p < 0)
    submitted = sum(o.intent.quantity for o in result.orders)
    exposure = [mv / eq for (_, mv), (_, eq) in zip(result.exposure_curve, result.equity_curve) if eq > 0]
    return RunMetrics(
        net_return=final / initial_cash - 1.0,
        max_drawdown=max_drawdown(_drawdown_series(result, series)),
        downside=downside_deviation(series),
        turnover=sum(f.quantity * f.price for f in result.fills) / initial_cash,
        fills=len(result.fills),
        orders=len(result.orders),
        rejected=sum(o.status is OrderStatus.REJECTED for o in result.orders),
        fees=result.final["fees_paid"],
        final_equity=final,
        sharpe=sharpe,
        sortino=sortino,
        round_trips=len(trips),
        losing_trades=sum(p < 0 for p in trips),
        profit_factor=gains / losses if trips and losses > 0 else None,
        win_rate=sum(p > 0 for p in trips) / len(trips) if trips else None,
        mean_trade_pnl=statistics.fmean(trips) if trips else None,
        fill_rate=sum(o.filled_qty for o in result.orders) / submitted if submitted else None,
        avg_exposure=statistics.fmean(exposure) if exposure else 0.0,
        peak_exposure=max(exposure, default=0.0),
        symbol_pnl=symbol_pnl(result),
    )


def _drawdown_series(result: RunResult, series: list[float]) -> list[float]:
    """Time-ordered union of curve samples and post-fill equity."""
    points = list(result.equity_curve)
    points += [(r["ts"], r["portfolio_equity"]) for r in result.audit if r.get("fill_qty")]
    if len(points) == len(result.equity_curve):
        return series
    points.sort(key=lambda p: p[0])
    return [series[0], *(e for _, e in points), series[-1]]


def latency_percentiles(samples_ns: list[int]) -> dict[str, float | None]:
    """Local decision-processing latency in microseconds (wall clock; machine-dependent)."""
    if not samples_ns:
        return {"p50_us": None, "p95_us": None, "p99_us": None, "samples": 0}
    s = sorted(samples_ns)

    def pct(q: float) -> float:
        return s[min(len(s) - 1, int(q * (len(s) - 1)))] / 1_000.0

    return {"p50_us": pct(0.50), "p95_us": pct(0.95), "p99_us": pct(0.99), "samples": len(s)}
