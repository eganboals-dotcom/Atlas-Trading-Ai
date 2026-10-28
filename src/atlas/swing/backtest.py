"""Daily-bar backtest with separate overnight and open-to-close legs, and per-trade costs.

Day t of a window: the overnight leg held from the close of t-1 earns ``x_on[t-1] * overnight[t]``, then the
day leg earns ``x_id[t] * intraday[t]``. Changing exposure costs ``cost_bps`` per unit at the open (from
``x_on[t-1]`` to ``x_id[t]``) and at the close (from ``x_id[t]`` to ``x_on[t]``). A window starts flat and is
sold at its last close, so windows can be scored independently. Symbols are equal-weighted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atlas.swing.data import Panel


@dataclass(frozen=True)
class Result:
    returns: np.ndarray  # daily portfolio return
    exposure: float  # mean fraction of capital in the market (both legs)
    trades: float  # entries + exits, per symbol


def run(p: Panel, x_id: np.ndarray, x_on: np.ndarray, start: int, end: int, cost_bps: float) -> Result:
    c = cost_bps / 1e4
    on, intra = p.overnight[start:end], p.intraday[start:end]
    xi = x_id[start:end]
    xo = x_on[start:end].copy()
    xo[-1] = 0.0  # sold at the window's last close
    prev = np.vstack([np.zeros((1, xo.shape[1])), xo[:-1]])
    open_turn = np.abs(xi - prev)
    close_turn = np.abs(xo - xi)
    # Legs compound at the portfolio level: x is in equal-weight slots, so the portfolio's leg return is the
    # mean over symbols (a symbol holding N/k slots carries weight 1/k).
    night, day = (prev * on).mean(axis=1), (xi * intra).mean(axis=1)
    cost = c * (open_turn + close_turn).mean(axis=1)
    trades = float(((open_turn > 1e-9).sum() + (close_turn > 1e-9).sum()) / xo.shape[1])
    return Result((1 + night) * (1 + day) - 1 - cost, float((xi.mean() + prev.mean()) / 2), trades)


def sharpe(r: np.ndarray) -> float:
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(252)) if sd > 1e-12 else 0.0


def max_drawdown(r: np.ndarray) -> float:
    eq = np.cumprod(1 + r)
    return float((1 - eq / np.maximum.accumulate(eq)).max()) if len(r) else 0.0


def summary(r: np.ndarray) -> dict[str, float]:
    return {"ann_return": float(np.prod(1 + r) ** (252 / max(1, len(r))) - 1), "ann_vol": float(r.std() * np.sqrt(252)),
            "sharpe": sharpe(r), "max_drawdown": max_drawdown(r), "days": len(r)}


def block_bootstrap(series: list[np.ndarray], stat, samples: int, block: int, seed: int) -> tuple[float, float]:
    """90% interval of ``stat(*series)`` over a moving-block bootstrap of aligned daily series (the same
    resampled days for every series, keeping volatility clusters)."""
    rng = np.random.default_rng(seed)
    n = len(series[0])
    k = int(np.ceil(n / block))
    vals = []
    for _ in range(samples):
        starts = rng.integers(0, n - block + 1, k)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        vals.append(stat(*(s[idx] for s in series)))
    lo, hi = np.percentile(vals, [5, 95])
    return float(lo), float(hi)
