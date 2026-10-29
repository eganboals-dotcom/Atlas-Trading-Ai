"""Swing families: each turns daily bars into an in-market signal s[t] in [0, 1] per symbol.

A signal for day t may only use what is known at the open of t: closes up to t-1 (``prev_close``) and the
open of t. ``mode`` then maps s to the two exposures of day t: 0 = held through (open-to-close and
overnight), 1 = overnight only (buy at the close, sell at the next open), 2 = open-to-close only.

Exposures are in units of an equal-weight slot (1 = this symbol's 1/N share of capital). Cross-sectional
families (``rotation``) put more in fewer symbols: holding the top k of N gives each one N/k, so the
portfolio is still fully invested at most. Families with ``legs`` set the two exposures directly.
"""

from __future__ import annotations

import warnings
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
        leaving = (held > 0) & (exit_[t] | ((held > max_days) if max_days else False))
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


def _rank_pick(score: np.ndarray, k: int, every: int, highest: bool) -> np.ndarray:
    """Equal weights in the ``k`` symbols with the highest (or lowest) score, picked every ``every`` days."""
    n = score.shape[1]
    k = min(k, n)

    def target(t: int) -> np.ndarray:
        sc = score[t]
        w = np.zeros(n)
        ok = ~np.isnan(sc)
        if ok.sum() < k:
            return w
        top = np.argsort(np.where(ok, -sc if highest else sc, np.inf), kind="stable")[:k]
        w[top] = n / k
        return w

    return _hold_between_rebalances(target, len(score), n, every)


def reversal(p: Panel, g: Genome) -> np.ndarray:
    """Short-term reversal: hold the ``top_k`` worst performers over the last ``lookback`` days."""
    pc = prev_close(p)
    with np.errstate(invalid="ignore"):
        score = pc / lag(pc, int(g["lookback"])) - 1
    return _rank_pick(score, int(g["top_k"]), int(g["rebalance"]), highest=False)


def low_vol(p: Panel, g: Genome) -> np.ndarray:
    """Low-volatility stocks: hold the ``top_k`` with the lowest daily volatility over ``vol_n`` days."""
    pc = prev_close(p)
    r = pc / lag(pc, 1) - 1
    m = sma(r, int(g["vol_n"]))
    vol = np.sqrt(np.clip(sma(r * r, int(g["vol_n"])) - m * m, 0.0, None))
    return _rank_pick(vol, int(g["top_k"]), int(g["rebalance"]), highest=False)


def risk_rotation(p: Panel, g: Genome) -> np.ndarray:
    """``rotation`` picks with two risk overlays, re-checked every ``overlay_every`` days and whenever the picks
    change: a pick below its ``trend_n``-day average (with ``trend_filter``) sits in cash, and with ``vol_scale``
    the whole book is scaled down so the picks' annualized volatility over the last ``vol_n`` days is at most
    ``target_vol``."""
    picks = rotation(p, g)
    pc = prev_close(p)
    n = pc.shape[1]
    r = np.nan_to_num(pc / lag(pc, 1) - 1)  # row t: close t-2 -> close t-1, known at the open of t
    trend = _trend_ok(p, g, pc)
    vol_n, every = int(g["vol_n"]), int(g["overlay_every"])
    w = np.zeros_like(pc)
    cur = np.zeros(n)
    for t in range(len(pc)):
        if t % every == 0 or not np.array_equal(picks[t], picks[t - 1]):
            cur = picks[t] * trend[t]
            if g["vol_scale"] and cur.sum() > 0 and t >= vol_n:
                vol = float((r[t - vol_n + 1:t + 1] @ (cur / n)).std() * np.sqrt(252))
                if vol > 1e-9:
                    cur = cur * min(1.0, g["target_vol"] / vol)
        w[t] = cur
    return w


def _entry_signal(p: Panel, g: Genome, pc: np.ndarray) -> np.ndarray:
    """Entry rule known at the open of t: 0 every day, 1 after a down close of at least ``drop_bps``, 2 on a
    gap down of at least ``drop_bps`` at the open."""
    drop = g["drop_bps"] / 1e4
    with np.errstate(invalid="ignore"):
        rule = {0.0: ~np.isnan(pc), 1.0: pc / lag(pc, 1) - 1 < -drop, 2.0: p.open / pc - 1 < -drop}[g["entry"]]
    return rule & _trend_ok(p, g, pc)


def bracket(p: Panel, g: Genome) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Buy at the open on an entry signal, then leave at +``tp_bps`` (a resting limit), at -``sl_bps`` (a stop),
    or at the close of the ``max_days``-th day, whichever comes first (``max_days`` 1 is a day trade). A gap
    through either level exits at that open. When a day touches both levels, the bar's shape decides which came
    first: a day that closed at or above its open is taken as open -> low -> high -> close (the stop first),
    otherwise open -> high -> low -> close (the target first). Checked against 1-minute paths of five symbols
    over ~900 days, this was within about 2 bps per trade of the true order, with no consistent lean; always
    taking the stop understated these bots by 2-20 bps per trade.

    Returns (open-to-close exposure, overnight exposure, exit return): where a position leaves intraday at a
    level, the exit return replaces that day's open-to-close return (NaN elsewhere).
    """
    pc = prev_close(p)
    enter = _entry_signal(p, g, pc)
    tp, sl, max_days = g["tp_bps"] / 1e4, g["sl_bps"] / 1e4, int(g["max_days"])
    days, n = pc.shape
    x_id, x_on = np.zeros((days, n)), np.zeros((days, n))
    exit_ret = np.full((days, n), np.nan)
    held = np.zeros(n, dtype=bool)
    entry_px = np.zeros(n)
    age = np.zeros(n, dtype=int)  # days in the position, counting today
    o, hi, lo = p.open, p.high, p.low
    for t in range(days):
        gap_out = held & ((o[t] >= entry_px * (1 + tp)) | (o[t] <= entry_px * (1 - sl)))
        held &= ~gap_out  # sold at this open: the overnight leg already earned the gap
        new = ~held & ~gap_out & enter[t]
        entry_px = np.where(new, o[t], entry_px)
        age = np.where(new, 0, age)
        held |= new
        age = np.where(held, age + 1, 0)
        x_id[t] = held
        stop, target = entry_px * (1 - sl), entry_px * (1 + tp)
        hit_sl, hit_tp = held & (lo[t] <= stop), held & (hi[t] >= target)
        stop_first = hit_sl & (~hit_tp | (p.close[t] >= o[t]))  # the day's outcome: used only to settle the exit
        with np.errstate(invalid="ignore", divide="ignore"):
            exit_ret[t] = np.where(stop_first, stop / o[t] - 1, np.where(hit_tp, target / o[t] - 1, np.nan))
        held &= ~(hit_sl | hit_tp) & (age < max_days)
        x_on[t] = held
    return x_id, x_on, exit_ret


def event_drift(p: Panel, g: Genome) -> np.ndarray:
    """Trade the days after a price event, a proxy for earnings and news without the news itself: yesterday
    gapped by at least ``gap_bps`` on at least ``vol_mult`` times its 20-day volume. ``direction`` +1 follows the
    gap (post-event drift), -1 fades it; long only, so a fade buys gap-downs. Held ``hold_days`` days."""
    with np.errstate(invalid="ignore", divide="ignore"):
        gap_prev = lag(p.open, 1) / lag(p.close, 2) - 1  # the gap of day t-1
        vol_ratio = lag(p.volume, 1) / lag(sma(p.volume, 20), 2)  # its volume vs the 20 days before it
        event = (g["direction"] * gap_prev > g["gap_bps"] / 1e4) & (vol_ratio > g["vol_mult"])
        if g["confirm"]:  # and the day closed in the gap's direction
            event &= g["direction"] * (lag(p.close, 1) / lag(p.open, 1) - 1) > 0
    pc = prev_close(p)
    return _hold_loop(event & _trend_ok(p, g, pc), np.zeros_like(event), int(g["hold_days"]))


def _feature(p: Panel, name: str) -> np.ndarray:
    """A news feature (see ``atlas.swing.news``); zeros when the panel has no news, so the family never trades."""
    return p.features.get(name, np.zeros(p.close.shape))


def news_gap(p: Panel, g: Genome) -> np.ndarray:
    """Gaps with and without news (after Chan, 2003: moves on news tend to continue, moves without news tend to
    reverse). At the open of t, a gap of at least ``gap_bps`` with at least ``min_articles`` articles since the
    last close is followed (``rule`` 0 buys gap-ups on news); a gap-down with no news at all is faded (``rule``
    1 buys it); ``rule`` 2 does both. Bought at the open of t and held ``hold_days`` days."""
    pc = prev_close(p)
    pre = _feature(p, "news_pre")  # articles from the last close to 09:15: known at the open
    with np.errstate(invalid="ignore"):
        gp = p.open / pc - 1
        follow = (gp > g["gap_bps"] / 1e4) & (pre >= g["min_articles"])
        fade = (gp < -g["gap_bps"] / 1e4) & (pre == 0)
    enter = {0.0: follow, 1.0: fade, 2.0: follow | fade}[g["rule"]] & _trend_ok(p, g, pc)
    return _hold_loop(enter, np.zeros_like(enter), int(g["hold_days"]))


def earnings_drift(p: Panel, g: Genome) -> np.ndarray:
    """Post-earnings drift with report dates taken from headlines: an earnings headline since the last close
    (or during the previous session) and a gap of at least ``gap_bps`` in ``direction`` (+1 up: follow good
    news; -1 down: buy the drop) at the open of t. Entered at that open, or a day later with ``delay``; held
    ``hold_days`` days."""
    pc = prev_close(p)
    earn = (_feature(p, "earn_pre") > 0) | (lag(_feature(p, "earn_day"), 1) > 0)
    with np.errstate(invalid="ignore"):
        event = earn & (g["direction"] * (p.open / pc - 1) > g["gap_bps"] / 1e4)
    if g["delay"]:
        event = np.nan_to_num(lag(event.astype(float), 1)) > 0
    return _hold_loop(event & _trend_ok(p, g, pc), np.zeros_like(event), int(g["hold_days"]))


def news_tone(p: Panel, g: Genome) -> np.ndarray:
    """Drift after tagged news known at the open of t (tags since the last close, or during the previous
    session): ``source`` 0 = analyst rating changes (upgrades minus downgrades), 1 = earnings surprises (EPS
    beats minus misses), 2 = both. ``side`` +1 buys good news, -1 buys bad news (betting on an overreaction).
    Entered at that open, or a day later with ``delay``; held ``hold_days`` days."""
    def known(tag: str) -> np.ndarray:
        return _feature(p, f"{tag}_pre") + np.nan_to_num(lag(_feature(p, f"{tag}_day"), 1))

    rating, surprise = known("up") - known("down"), known("beat") - known("miss")
    score = {0.0: rating, 1.0: surprise, 2.0: rating + surprise}[g["source"]]
    event = g["side"] * score > 0
    if g["delay"]:
        event = np.nan_to_num(lag(event.astype(float), 1)) > 0
    pc = prev_close(p)
    return _hold_loop(event & _trend_ok(p, g, pc), np.zeros_like(event), int(g["hold_days"]))


def brief_drift(p: Panel, g: Genome) -> np.ndarray:
    """Trade the news desk's briefing (see ``atlas.news.desk``): buy at the open of t when the score known at
    that open is at least ``thr`` in ``side``'s direction (+1 good news, -1 bad news), held ``hold_days``."""
    score = _feature(p, "brief_score")
    pc = prev_close(p)
    enter = (g["side"] * score >= g["thr"]) & (score != 0) & _trend_ok(p, g, pc)
    return _hold_loop(enter, np.zeros_like(enter), int(g["hold_days"]))


def brief_tilt(p: Panel, g: Genome) -> np.ndarray:
    """Hold every symbol like Hold, but tilted by the briefing: over the last ``window`` opens, symbols whose
    summed score is at most -``thr`` are dropped and those at least +``thr`` get ``1 + tilt`` times the weight;
    always fully invested. Weights are reset every ``rebalance`` days."""
    score = _feature(p, "brief_score")
    recent = np.nan_to_num(sma(score, int(g["window"]))) * int(g["window"])
    pc = prev_close(p)
    n = pc.shape[1]

    def target(t: int) -> np.ndarray:
        w = np.where(recent[t] <= -g["thr"], 0.0, np.where(recent[t] >= g["thr"], 1 + g["tilt"], 1.0))
        w = w * ~np.isnan(pc[t])
        return w * n / w.sum() if w.sum() > 0 else w

    return _hold_between_rebalances(target, len(pc), n, int(g["rebalance"]))


def _xs_rank(x: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank per day (0 worst .. 1 best), NaN stays NaN."""
    out = np.full_like(x, np.nan)
    for t in range(len(x)):
        ok = ~np.isnan(x[t])
        if ok.sum() > 1:
            out[t, ok] = np.argsort(np.argsort(x[t, ok])) / (ok.sum() - 1)
    return out


FUND_FACTORS = {0.0: "gross profitability", 1.0: "return on assets", 2.0: "book-to-market", 3.0: "earnings yield",
                4.0: "quality + value", 5.0: "earnings surprise", 6.0: "earnings surprise (news)"}


def fund_rank(p: Panel, g: Genome) -> np.ndarray:
    """Hold the ``top_k`` companies by a fundamental factor (see ``atlas.swing.fundamentals``), equal weight,
    re-picked every ``rebalance`` days. ``factor`` 4 averages the ranks of quality (gross profitability, else
    return on assets) and value (earnings yield), in the spirit of Greenblatt's "magic formula"; ``factor`` 5
    picks the largest recent earnings surprises (companies without one in the last quarter are skipped), and
    ``factor`` 6 the same from headlines (reported EPS vs the analysts' estimate, from the announcement day)."""
    feat = {k: _feature(p, k) if k in p.features else np.full(p.close.shape, np.nan)
            for k in ("fund_gpa", "fund_roa", "fund_bm", "fund_ep", "fund_sue", "news_surprise")}
    f = g["factor"]
    if f == 4.0:
        quality = np.where(np.isnan(feat["fund_gpa"]), feat["fund_roa"], feat["fund_gpa"])
        score = (_xs_rank(quality) + _xs_rank(feat["fund_ep"])) / 2
    else:
        score = feat[{0.0: "fund_gpa", 1.0: "fund_roa", 2.0: "fund_bm", 3.0: "fund_ep", 5.0: "fund_sue",
                     6.0: "news_surprise"}[f]]
    pc = prev_close(p)
    score = np.where(np.isnan(pc), np.nan, score)
    return _rank_pick(score, int(g["top_k"]), int(g["rebalance"]), highest=True)


TILT_FACTORS = {0.0: "quality", 1.0: "momentum", 2.0: "low volatility", 3.0: "quality + momentum + low volatility",
                4.0: "earnings surprise", 5.0: "insider buying", 6.0: "low short interest",
                7.0: "earnings surprise (news)", 8.0: "quality + earnings surprise + low short interest"}
TILT_FEATURES = ("tilt_quality", "tilt_momentum", "tilt_lowvol")  # factor 3 averages exactly these three
EXTRA_TILT = {4.0: "tilt_earnings", 5.0: "tilt_insider", 6.0: "tilt_short", 7.0: "tilt_news_earnings"}
PASSED_TILTS = ("tilt_quality", "tilt_earnings", "tilt_short")  # factor 8: the three that passed in v8-v11
INSIDER_MIN_BUYERS = 2  # a "cluster": at least two different insiders buying on the open market


def _extra_tilt(p: Panel, key: str) -> np.ndarray:
    """Scores for the event-like tilts; NaN (equal weight) where there is no information."""
    pc = prev_close(p)
    nan = np.full(pc.shape, np.nan)
    if key == "tilt_earnings":
        return _xs_rank(np.where(np.isnan(pc), np.nan, p.features.get("fund_sue", nan)))
    if key == "tilt_insider":  # flagged names get the top score, the rest stay at equal weight
        n = p.features.get("ins_buyers", nan)
        return np.where(np.isnan(pc) | np.isnan(n), np.nan, np.where(n >= INSIDER_MIN_BUYERS, 1.0, 0.5))
    if key == "tilt_short":  # fewest days-to-cover best; no data before June 2021
        return _xs_rank(np.where(np.isnan(pc), np.nan, -p.features.get("short_dtc", nan)))
    if key == "tilt_news_earnings":
        return _xs_rank(np.where(np.isnan(pc), np.nan, p.features.get("news_surprise", nan)))
    raise KeyError(key)


def tilt_features(p: Panel) -> dict[str, np.ndarray]:
    """Cross-sectional ranks (0 worst .. 1 best) known at each open: quality (gross profitability, else return on
    assets, from filings already public), momentum (return from 12 to 1 months ago) and low volatility (lower
    daily volatility over the last year), all from closes up to t-1."""
    pc = prev_close(p)
    with np.errstate(invalid="ignore", divide="ignore"):
        mom = lag(pc, 21) / lag(pc, 252) - 1
        r = pc / lag(pc, 1) - 1
        vol = np.sqrt(np.clip(sma(r * r, 252) - sma(r, 252) ** 2, 0, None))
    gpa = p.features.get("fund_gpa", np.full(pc.shape, np.nan))
    roa = p.features.get("fund_roa", np.full(pc.shape, np.nan))
    quality = np.where(np.isnan(gpa), roa, gpa)
    out = {"tilt_quality": _xs_rank(np.where(np.isnan(pc), np.nan, quality)), "tilt_momentum": _xs_rank(mom),
           "tilt_lowvol": _xs_rank(-vol)}
    for key, src in (("tilt_earnings", "fund_sue"), ("tilt_insider", "ins_buyers"), ("tilt_short", "short_dtc"),
                     ("tilt_news_earnings", "news_surprise")):
        if src in p.features:
            out[key] = _extra_tilt(p, key)
    return out


def tilt(p: Panel, g: Genome) -> np.ndarray:
    """Always fully invested, like Hold, but each symbol's weight leans with a factor's rank: from 1 - ``tilt``
    to 1 + ``tilt`` times equal weight (best ranked highest). A symbol without a value keeps equal weight.
    Re-weighted every ``rebalance`` days. ``factor`` 3 uses the mean of the three ranks (see ``tilt_features``);
    4-7 lean on earnings surprises, insider buying clusters, low short interest and headline earnings surprises;
    8 on the mean rank of the three that passed their tests (quality, earnings surprise, low short interest; a
    rank missing on a day, like short interest before June 2021, is left out of that day's mean)."""
    f = g["factor"]
    if f in EXTRA_TILT:
        key = EXTRA_TILT[f]
        feats = {key: p.features[key] if key in p.features else _extra_tilt(p, key)}
    elif f == 8.0:
        have = p.features if all(k in p.features for k in PASSED_TILTS) else {**tilt_features(p), **p.features}
        feats = {k: have[k] for k in PASSED_TILTS if k in have}
    elif all(k in p.features for k in TILT_FEATURES):
        feats = {k: p.features[k] for k in TILT_FEATURES}
    else:
        feats = tilt_features(p)
    if f in EXTRA_TILT:
        score = feats[EXTRA_TILT[f]]
    elif f in (3.0, 8.0):
        keys = TILT_FEATURES if f == 3.0 else PASSED_TILTS
        stack = np.stack([feats[k] if k in feats else np.full(p.close.shape, np.nan) for k in keys])
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN days: no rank, equal weight
            score = np.nanmean(stack, axis=0)
        score = _xs_rank(score)
    else:
        score = feats[TILT_FEATURES[int(f)]]
    listed = ~np.isnan(prev_close(p))
    n = p.close.shape[1]

    def target(t: int) -> np.ndarray:
        w = np.where(listed[t], 1.0, 0.0)
        ok = listed[t] & ~np.isnan(score[t])
        w[ok] = 1 + g["tilt"] * (2 * score[t, ok] - 1)
        return w

    every = int(g["rebalance"])
    w = _hold_between_rebalances(target, len(p.dates), n, every)
    last_pick = listed[np.arange(len(p.dates)) // every * every]
    w = np.where(listed & ~last_pick, 1.0, np.where(listed, w, 0.0))  # listed since the last pick: equal weight
    total = w.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):  # invested like Hold every day, only the mix changes
        return np.where(total > 0, w * listed.sum(axis=1, keepdims=True) / total, 0.0)


def insider_buy(p: Panel, g: Genome) -> np.ndarray:
    """Hold, equal weight, every company where at least ``min_buyers`` different directors or officers bought
    shares on the open market in the last 90 days (filings already public; see ``atlas.swing.insiders``)."""
    n = p.features.get("ins_buyers", np.zeros(p.close.shape))
    return ((n >= g["min_buyers"]) & ~np.isnan(prev_close(p))).astype(float)


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
    legs: Callable[[Panel, Genome], tuple[np.ndarray, ...]] | None = None


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
    Family("risk_rotation", {"lookback": ParamSpec("int", 63, 252, log=True), "skip": ParamSpec("int", 0, 21),
                             "top_k": ParamSpec("int", 1, 8), "rebalance": ParamSpec("int", 5, 63, log=True),
                             "abs_filter": ParamSpec("choice", choices=(0.0, 1.0)), **_FILTER,
                             "vol_scale": ParamSpec("choice", choices=(0.0, 1.0)),
                             "vol_n": ParamSpec("int", 10, 120, log=True), "target_vol": ParamSpec("float", 0.05, 0.4),
                             "overlay_every": ParamSpec("choice", choices=(1.0, 5.0, 21.0))}, risk_rotation,
           fixed_mode=0.0),
    Family("reversal", {**_MODE, "lookback": ParamSpec("int", 2, 21), "top_k": ParamSpec("int", 1, 10),
                        "rebalance": ParamSpec("int", 1, 21)}, reversal),
    Family("low_vol", {**_MODE, "vol_n": ParamSpec("int", 20, 252, log=True), "top_k": ParamSpec("int", 3, 15),
                       "rebalance": ParamSpec("int", 5, 63, log=True)}, low_vol),
    # targets of +0.5% to +3% per trade
    Family("bracket", {**_FILTER, "entry": ParamSpec("choice", choices=(0.0, 1.0, 2.0)),
                       "drop_bps": ParamSpec("float", 25.0, 500.0, log=True),
                       "tp_bps": ParamSpec("float", 50.0, 300.0, log=True),
                       "sl_bps": ParamSpec("float", 50.0, 800.0, log=True), "max_days": ParamSpec("int", 1, 20)},
           legs=bracket),
    # swing targets: +3% to +20% per trade, held up to ~6 months
    Family("wide_bracket", {**_FILTER, "entry": ParamSpec("choice", choices=(0.0, 1.0, 2.0)),
                            "drop_bps": ParamSpec("float", 25.0, 500.0, log=True),
                            "tp_bps": ParamSpec("float", 300.0, 2000.0, log=True),
                            "sl_bps": ParamSpec("float", 200.0, 1500.0, log=True),
                            "max_days": ParamSpec("int", 5, 126, log=True)}, legs=bracket),
    Family("event_drift", {**_MODE, **_FILTER, "direction": ParamSpec("choice", choices=(-1.0, 1.0)),
                           "gap_bps": ParamSpec("float", 100.0, 1000.0, log=True),
                           "vol_mult": ParamSpec("float", 1.0, 5.0), "confirm": ParamSpec("choice", choices=(0.0, 1.0)),
                           "hold_days": ParamSpec("int", 1, 60, log=True)}, event_drift),
    # need news features on the panel (``news_dir`` in the config); without them they never trade
    Family("news_gap", {**_FILTER, "rule": ParamSpec("choice", choices=(0.0, 1.0, 2.0)),
                        "gap_bps": ParamSpec("float", 50.0, 800.0, log=True), "min_articles": ParamSpec("int", 1, 5),
                        "hold_days": ParamSpec("int", 1, 40, log=True)}, news_gap, fixed_mode=0.0),
    Family("earnings_drift", {**_FILTER, "direction": ParamSpec("choice", choices=(-1.0, 1.0)),
                              "gap_bps": ParamSpec("float", 0.0, 800.0), "delay": ParamSpec("choice", choices=(0.0, 1.0)),
                              "hold_days": ParamSpec("int", 1, 60, log=True)}, earnings_drift, fixed_mode=0.0),
    Family("brief_drift", {**_FILTER, "side": ParamSpec("choice", choices=(-1.0, 1.0)),
                           "thr": ParamSpec("float", 0.2, 3.0, log=True), "hold_days": ParamSpec("int", 1, 60, log=True)},
           brief_drift, fixed_mode=0.0),
    Family("brief_tilt", {"window": ParamSpec("int", 1, 63, log=True), "thr": ParamSpec("float", 0.2, 3.0, log=True),
                          "tilt": ParamSpec("float", 0.0, 2.0), "rebalance": ParamSpec("int", 1, 21)},
           brief_tilt, fixed_mode=0.0),
    # needs fundamentals on the panel (``fundamentals_dir`` in the config)
    Family("fund_rank", {"factor": ParamSpec("choice", choices=tuple(FUND_FACTORS)), "top_k": ParamSpec("int", 3, 15),
                         "rebalance": ParamSpec("int", 5, 126, log=True)}, fund_rank, fixed_mode=0.0),
    Family("tilt", {"factor": ParamSpec("choice", choices=tuple(TILT_FACTORS)), "tilt": ParamSpec("float", 0.0, 1.0),
                    "rebalance": ParamSpec("int", 5, 126, log=True)}, tilt, fixed_mode=0.0),
    Family("insider_buy", {"min_buyers": ParamSpec("int", 1, 4)}, insider_buy, fixed_mode=0.0),
    Family("news_tone", {**_FILTER, "source": ParamSpec("choice", choices=(0.0, 1.0, 2.0)),
                         "side": ParamSpec("choice", choices=(-1.0, 1.0)), "delay": ParamSpec("choice", choices=(0.0, 1.0)),
                         "hold_days": ParamSpec("int", 1, 60, log=True)}, news_tone, fixed_mode=0.0),
)}

HOLD = ("session", {"mode": 0.0, "trend_filter": 0.0, "trend_n": 200})


def exposures(p: Panel, family: str, g: Genome) -> tuple[np.ndarray, ...]:
    """(open-to-close exposure, overnight exposure) for every day, each (days, symbols), in equal-weight slots;
    families that exit intraday at a price level add a third array of exit returns (see ``bracket``)."""
    fam = FAMILIES[family]
    if fam.legs is not None:
        x_id, x_on, *extra = fam.legs(p, g)
        return (np.nan_to_num(x_id, nan=0.0), np.nan_to_num(x_on, nan=0.0), *extra)
    s = np.nan_to_num(fam.signal(p, g), nan=0.0)
    mode = fam.fixed_mode if fam.fixed_mode is not None else g["mode"]
    zero = np.zeros_like(s)
    return (zero if mode == 1 else s), (zero if mode == 2 else s)
