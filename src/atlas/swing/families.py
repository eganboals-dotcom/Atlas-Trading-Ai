"""Swing families: each turns daily bars into an in-market signal s[t] in [0, 1] per symbol.

A signal for day t may only use what is known at the open of t: closes up to t-1 (``prev_close``) and the
open of t. ``mode`` then maps s to the two exposures of day t: 0 = held through (open-to-close and
overnight), 1 = overnight only (buy at the close, sell at the next open), 2 = open-to-close only.

Exposures are in units of an equal-weight slot (1 = this symbol's 1/N share of capital). Cross-sectional
families (``rotation``) put more in fewer symbols: holding the top k of N gives each one N/k, so the
portfolio is still fully invested at most. Families with ``legs`` set the two exposures directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from atlas.agents.genome import Genome, ParamSpec
from atlas.swing.data import Panel

MODES = (0.0, 1.0, 2.0)
_MODE = {"mode": ParamSpec("choice", choices=MODES)}
_FILTER = {"trend_filter": ParamSpec("choice", choices=(0.0, 1.0)), "trend_n": ParamSpec("int", 50, 250)}


def prev_close(p: Panel) -> np.ndarray:
    c = np.full_like(p.close, np.nan)
    c[1:] = p.close[:-1]
    return c


def sma(x: np.ndarray, n: int) -> np.ndarray:
    """Trailing mean over n rows (NaN until n valid values; NaNs at the start are skipped)."""
    n = max(1, int(n))
    out = np.full_like(x, np.nan)
    v = np.where(np.isnan(x), 0.0, x)
    cs = np.cumsum(v, axis=0)
    cnt = np.cumsum(~np.isnan(x), axis=0)
    s = cs.copy()
    s[n:] -= cs[:-n]
    k = cnt.copy()
    k[n:] -= cnt[:-n]
    ok = k == n
    out[ok] = s[ok] / n
    return out


def lag(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if n < len(x):
        out[n:] = x[:-n] if n else x
    return out


def _trend_ok(p: Panel, g: Genome, pc: np.ndarray) -> np.ndarray:
    if not g.get("trend_filter"):
        return np.ones_like(pc, dtype=bool)
    with np.errstate(invalid="ignore"):
        return pc > sma(pc, int(g["trend_n"]))


def trend(p: Panel, g: Genome) -> np.ndarray:
    pc = prev_close(p)
    fast = int(g["fast"])
    slow = max(fast + 1, int(round(fast * g["slow_mult"])))
    with np.errstate(invalid="ignore"):
        return (sma(pc, fast) > sma(pc, slow) * (1 + g["band_bps"] / 1e4)).astype(float)


def tsmom(p: Panel, g: Genome) -> np.ndarray:
    pc = prev_close(p)
    with np.errstate(invalid="ignore"):
        return (pc / lag(pc, int(g["lookback"])) - 1 > g["thr_bps"] / 1e4).astype(float)


def _hold_loop(enter: np.ndarray, exit_: np.ndarray, max_days: int | None) -> np.ndarray:
    """Stateful in/out: enter on ``enter``, leave on ``exit_`` or after ``max_days`` days in."""
    s = np.zeros(enter.shape)
    held = np.zeros(enter.shape[1], dtype=int)  # days in the position, 0 = flat
    for t in range(enter.shape[0]):
        leaving = (held > 0) & (exit_[t] | ((held >= max_days) if max_days else False))
        held[leaving] = 0
        held[(held == 0) & ~leaving & enter[t]] = 1
        s[t] = held > 0
        held[held > 0] += 1
    return s


def dip(p: Panel, g: Genome) -> np.ndarray:
    pc = prev_close(p)
    with np.errstate(invalid="ignore"):
        drop = pc / lag(pc, int(g["drop_n"])) - 1 < -g["drop_bps"] / 1e4
        enter = drop & _trend_ok(p, g, pc)
        exit_ = pc > sma(pc, int(g["exit_n"]))
    return _hold_loop(enter, exit_, int(g["hold_days"]))


def breakout(p: Panel, g: Genome) -> np.ndarray:
    pc = prev_close(p)
    n_in, n_out = int(g["entry_n"]), int(g["exit_n"])
    hi = np.full_like(pc, np.nan)
    lo = np.full_like(pc, np.nan)
    for t in range(1, len(pc)):  # extremes of the closes before t-1
        a, b = max(0, t - 1 - n_in), t - 1
        if b - a == n_in:
            hi[t] = np.nanmax(pc[a:b], axis=0)
        a2 = max(0, t - 1 - n_out)
        if b - a2 == n_out:
            lo[t] = np.nanmin(pc[a2:b], axis=0)
    with np.errstate(invalid="ignore"):
        return _hold_loop(pc > hi, pc < lo, None)


def vol_target(p: Panel, g: Genome) -> np.ndarray:
    pc = prev_close(p)
    r = pc / lag(pc, 1) - 1
    m = sma(r, int(g["vol_n"]))
    var = sma(r * r, int(g["vol_n"])) - m * m
    vol = np.sqrt(np.clip(var, 1e-12, None) * 252)
    s = np.clip(g["target_vol"] / vol, 0.0, 1.0)
    s = np.where(np.isnan(s), 0.0, s)
    return s * _trend_ok(p, g, pc)


def session(p: Panel, g: Genome) -> np.ndarray:
    """Always in (optionally only above a trend line): with ``mode`` this is Hold, overnight-only or day-only."""
    pc = prev_close(p)
    return _trend_ok(p, g, pc).astype(float) * ~np.isnan(pc)


def gap(p: Panel, g: Genome) -> np.ndarray:
    """Open-to-close trade on the morning gap: buy gap-downs (direction -1) or gap-ups (+1)."""
    pc = prev_close(p)
    with np.errstate(invalid="ignore"):
        gp = p.open / pc - 1
        return ((g["direction"] * gp > g["gap_bps"] / 1e4) & _trend_ok(p, g, pc)).astype(float)


def _hold_between_rebalances(target: Callable[[int], np.ndarray], days: int, n: int, every: int) -> np.ndarray:
    """Weights recomputed on every ``every``-th day (a fixed grid from the first day) and held in between."""
    w = np.zeros((days, n))
    cur = np.zeros(n)
    for t in range(days):
        if t % every == 0:
            cur = target(t)
        w[t] = cur
    return w


def rotation(p: Panel, g: Genome) -> np.ndarray:
    """Hold the top ``top_k`` symbols by return over ``lookback`` days (ending ``skip`` days ago), rebalanced
    every ``rebalance`` days; with ``abs_filter`` a slot whose symbol is down over the lookback stays in cash."""
    pc = prev_close(p)
    n = pc.shape[1]
    k = min(int(g["top_k"]), n)
    skip, look = int(g["skip"]), int(g["lookback"])
    with np.errstate(invalid="ignore"):
        score = lag(pc, skip) / lag(pc, skip + look) - 1

    def target(t: int) -> np.ndarray:
        sc = score[t]
        w = np.zeros(n)
        ok = ~np.isnan(sc)
        if ok.sum() < k:
            return w
        top = np.argsort(np.where(ok, -sc, np.inf), kind="stable")[:k]
        if g["abs_filter"]:
            top = top[sc[top] > 0]
        w[top] = n / k
        return w

    return _hold_between_rebalances(target, len(pc), n, int(g["rebalance"]))


def leg_select(p: Panel, g: Genome) -> tuple[np.ndarray, np.ndarray]:
    """Per symbol, hold each leg (overnight, open-to-close) while its own trailing mean return clears a bar."""
    look, every = int(g["lookback"]), int(g["rebalance"])
    m_on = sma(lag(p.overnight, 1), look)  # legs up to t-1, both known at the open of t
    m_id = sma(lag(p.intraday, 1), look)
    with np.errstate(invalid="ignore"):
        x_on, x_id = (m_on > g["on_bps"] / 1e4).astype(float), (m_id > g["id_bps"] / 1e4).astype(float)
    n = x_on.shape[1]
    return (_hold_between_rebalances(lambda t: x_id[t], len(x_id), n, every),
            _hold_between_rebalances(lambda t: x_on[t], len(x_on), n, every))


def month_position(dates: tuple) -> tuple[np.ndarray, np.ndarray]:
    """(trading day of the month counted from 1, trading days left in the month counting the day itself).
    The exchange calendar is published in advance, so both are known before the day starts. The last
    month of a truncated panel looks shorter than it is; walk-forward windows never reach it."""
    months = [(d.year, d.month) for d in dates]
    first = np.zeros(len(dates), dtype=int)
    left = np.zeros(len(dates), dtype=int)
    i = 0
    while i < len(dates):
        j = i
        while j < len(dates) and months[j] == months[i]:
            j += 1
        first[i:j] = np.arange(1, j - i + 1)
        left[i:j] = np.arange(j - i, 0, -1)
        i = j
    return first, left


def calendar(p: Panel, g: Genome) -> np.ndarray:
    """In the market over the turn of the month: the last ``before`` and the first ``after`` trading days."""
    first, left = month_position(p.dates)
    on = (left <= int(g["before"])) | (first <= int(g["after"]))
    pc = prev_close(p)
    return on[:, None] * _trend_ok(p, g, pc).astype(float) * ~np.isnan(pc)


@dataclass(frozen=True)
class Family:
    name: str
    space: dict[str, ParamSpec]
    signal: Callable[[Panel, Genome], np.ndarray] | None = None
    fixed_mode: float | None = None
    legs: Callable[[Panel, Genome], tuple[np.ndarray, np.ndarray]] | None = None


FAMILIES: dict[str, Family] = {f.name: f for f in (
    Family("trend", {**_MODE, "fast": ParamSpec("int", 5, 100, log=True), "slow_mult": ParamSpec("float", 1.5, 6.0),
                     "band_bps": ParamSpec("float", 0.0, 200.0)}, trend),
    Family("tsmom", {**_MODE, "lookback": ParamSpec("int", 10, 250, log=True),
                     "thr_bps": ParamSpec("float", -500.0, 1000.0)}, tsmom),
    Family("dip", {**_MODE, **_FILTER, "drop_n": ParamSpec("int", 1, 10), "drop_bps": ParamSpec("float", 50.0, 1500.0, log=True),
                   "hold_days": ParamSpec("int", 1, 20), "exit_n": ParamSpec("int", 3, 20)}, dip),
    Family("breakout", {**_MODE, "entry_n": ParamSpec("int", 10, 250, log=True),
                        "exit_n": ParamSpec("int", 5, 100, log=True)}, breakout),
    Family("vol_target", {**_MODE, **_FILTER, "vol_n": ParamSpec("int", 10, 120, log=True),
                          "target_vol": ParamSpec("float", 0.05, 0.4)}, vol_target),
    Family("session", {**_MODE, **_FILTER}, session),
    Family("gap", {**_FILTER, "gap_bps": ParamSpec("float", 20.0, 500.0, log=True),
                   "direction": ParamSpec("choice", choices=(-1.0, 1.0))}, gap, fixed_mode=2.0),
    Family("rotation", {**_MODE, "lookback": ParamSpec("int", 20, 252, log=True), "skip": ParamSpec("int", 0, 21),
                        "top_k": ParamSpec("int", 1, 8), "rebalance": ParamSpec("int", 5, 63, log=True),
                        "abs_filter": ParamSpec("choice", choices=(0.0, 1.0))}, rotation),
    Family("leg_select", {"lookback": ParamSpec("int", 20, 252, log=True), "on_bps": ParamSpec("float", -5.0, 10.0),
                          "id_bps": ParamSpec("float", -5.0, 10.0), "rebalance": ParamSpec("int", 5, 63, log=True)},
           legs=leg_select),
    Family("calendar", {**_MODE, **_FILTER, "before": ParamSpec("int", 0, 5), "after": ParamSpec("int", 0, 5)},
           calendar),
)}

HOLD = ("session", {"mode": 0.0, "trend_filter": 0.0, "trend_n": 200})


def exposures(p: Panel, family: str, g: Genome) -> tuple[np.ndarray, np.ndarray]:
    """(open-to-close exposure, overnight exposure) for every day, each (days, symbols), in equal-weight slots."""
    fam = FAMILIES[family]
    if fam.legs is not None:
        x_id, x_on = fam.legs(p, g)
        return np.nan_to_num(x_id, nan=0.0), np.nan_to_num(x_on, nan=0.0)
    s = np.nan_to_num(fam.signal(p, g), nan=0.0)
    mode = fam.fixed_mode if fam.fixed_mode is not None else g["mode"]
    zero = np.zeros_like(s)
    return (zero if mode == 1 else s), (zero if mode == 2 else s)
