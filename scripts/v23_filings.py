"""Four slow rules from company filings (see configs/filings_v23.yaml, which holds the pre-registered tests), on the
unseen days. fetch reads operating cash flow and diluted shares from the SEC company files. run writes
artifacts/reports/swing/v23_filings.json.

    .venv/Scripts/python.exe scripts/v23_filings.py fetch
    .venv/Scripts/python.exe scripts/v23_filings.py run
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import yaml

import v19_pairs as pairs
import v22_bottleneck as bottleneck
from atlas.config import load_config
from atlas.feeds import sec
from atlas.swing.backtest import run, simulate, summary
from atlas.swing.data import load_panel
from atlas.swing.families import HOLD, exposures
from atlas.swing.fundamentals import SCHEMA, _rows, fundamental_features
from atlas.swing.walkforward import SwingConfig

cfg = yaml.safe_load(Path("configs/filings_v23.yaml").read_text(encoding="utf-8"))
OUT = Path("artifacts/reports/swing/v23_filings.json")
CFO = [("us-gaap", "NetCashProvidedByUsedInOperatingActivities"), ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")]
DILUTED = [("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding")]
RULES = ("buybacks", "asset_growth", "earnings_quality", "quality_value")


def extract_extra(facts: dict) -> pl.DataFrame:
    rows = [("cfo", *r) for r in _rows(facts, CFO, "USD", annual=True, instant=False)]
    rows += [("diluted", *r) for r in _rows(facts, DILUTED, "shares", annual=True, instant=False)]
    return pl.DataFrame(rows, schema=SCHEMA, orient="row") if rows else pl.DataFrame(schema=SCHEMA)


def cmd_fetch() -> None:
    symbols = load_config(Path(cfg["swing_config"]), SwingConfig).symbols
    out = Path(cfg["extra_dir"])
    out.mkdir(parents=True, exist_ok=True)
    get = sec.http_get(sec.user_agent(Path(".env")))
    ciks = sec.ciks(get)
    for sym in symbols:
        path = out / f"{sym}.parquet"
        cik = ciks.get(sym, sec.DELISTED_CIKS.get(sym))
        if path.exists() or cik is None:
            continue
        filers = (*sec.EARLIER_CIKS.get(sym, ()), cik)
        pl.concat([extract_extra(sec.company_facts(get, c)) for c in filers]).unique(maintain_order=True).write_parquet(path)
    print("stored", len(list(out.glob("*.parquet"))), "of", len(symbols), flush=True)


def latest_two(rows: pl.DataFrame, item: str, d: date) -> tuple[date, float, float] | None:
    """Fiscal year end, its value and the year before's, as last filed before day d."""
    lo, hi = cfg["prior_year_days"]
    vals = dict(rows.filter((pl.col("item") == item) & (pl.col("filed") < d)).sort("filed").select("end", "val").rows())
    if not vals:
        return None
    end = max(vals)
    prev = [e for e in vals if lo <= (end - e).days <= hi]
    if (d - end).days > cfg["max_age_days"] or not prev:
        return None
    return end, vals[end], vals[min(prev, key=lambda e: abs((end - e).days - 365))]


def filing_scores(rows: pl.DataFrame, d: date, has_gross_profit: bool) -> dict[str, float]:
    """The scores of rules I, J and K a company can have on day d. Higher is better."""
    out = {}
    shares = latest_two(rows, "diluted", d)
    if shares and shares[1] > 0 and shares[2] > 0:
        out["buybacks"] = -(shares[1] / shares[2] - 1)
    assets = latest_two(rows, "assets", d)
    if assets and assets[1] > 0 and assets[2] > 0:
        out["asset_growth"] = -(assets[1] / assets[2] - 1)
        income, cash = latest_two(rows, "net_income", d), latest_two(rows, "cfo", d)
        if has_gross_profit and income and cash and income[0] == cash[0] == assets[0]:
            out["earnings_quality"] = -(income[1] - cash[1]) / ((assets[1] + assets[2]) / 2)
    return out


def percentile(x: np.ndarray) -> np.ndarray:
    """0 worst to 1 best among finite values, NaN stays."""
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    out[ok] = np.argsort(np.argsort(x[ok], kind="stable"), kind="stable") / max(1, ok.sum() - 1)
    return out


def top(score: np.ndarray, share: float, highest: bool = True) -> list[int]:
    ok = np.nonzero(np.isfinite(score))[0]
    order = ok[np.argsort(-score[ok] if highest else score[ok], kind="stable")]
    return order[: max(1, int(len(ok) * share))].tolist()


def by_sector(score: np.ndarray, group: np.ndarray) -> list[int]:
    picks: list[int] = []
    for g in sorted(set(group.tolist())):
        inside = np.where(group == g, score, np.nan)
        if np.isfinite(inside).sum() >= cfg["min_group"]:
            picks += top(inside, cfg["top_share"])
    return picks


def stats(r: np.ndarray, rf: np.ndarray) -> dict:
    return {k: round(v, 4) for k, v in summary(r, rf).items()}


def beats(a: dict, b: dict) -> bool:
    return bool(a["sharpe"] > b["sharpe"] and a["ann_return"] >= b["ann_return"])


def cmd_run() -> None:
    swing = load_config(Path(cfg["swing_config"]), SwingConfig)
    m = pairs.market(cfg["swing_config"])
    p, v0, end = m.p, m.v0, m.end
    n, days = p.close.shape[1], len(p.dates)
    rf = m.cash[v0:end]
    e = load_panel(Path(bottleneck.cfg["etf_dir"]), bottleneck.cfg["etfs"] + [bottleneck.cfg["market"]])
    at = {d: i for i, d in enumerate(e.dates)}
    idx = [at[d] for d in p.dates]
    etf, mkt = e.close[idx, :-1], e.close[idx, -1]
    raw = load_panel(swing.raw_dataset_dir, swing.symbols)
    raw_at = {d: i for i, d in enumerate(raw.dates)}
    feats = fundamental_features(p, Path(cfg["fundamentals_dir"]), raw.close[[raw_at[d] for d in p.dates]])
    rows = {}
    for s in p.symbols:
        files = [Path(cfg[k]) / f"{s}.parquet" for k in ("fundamentals_dir", "extra_dir")]
        rows[s] = pl.concat([pl.read_parquet(f) for f in files if f.exists()]) if any(f.exists() for f in files) else pl.DataFrame(schema=SCHEMA)
    names = (*RULES, "mix")
    x = {f"{k}_{side}": np.zeros((days, n)) for k in names for side in ("top", "bottom", "sector")}
    eligible: dict[str, list[int]] = {k: [] for k in names}
    corr: dict[str, list[float]] = {k: [] for k in RULES}
    everyone = list(range(n))

    def hold_until(key: str, t: int, picks: list[int]) -> None:
        x[key][t: t + cfg["every"]] = 0.0
        x[key][t: t + cfg["every"], picks] = n / len(picks)

    for t in range(v0, end, cfg["every"]):
        d = p.dates[t]
        gpa, bm = feats["fund_gpa"][t], feats["fund_bm"][t]
        score = {k: np.full(n, np.nan) for k in RULES}
        for j, s in enumerate(p.symbols):
            for k, v in filing_scores(rows[s], d, bool(np.isfinite(gpa[j]))).items():
                score[k][j] = v
        both = np.isfinite(gpa) & np.isfinite(bm)
        score["quality_value"][both] = (percentile(np.where(both, gpa, np.nan))[both] + percentile(np.where(both, bm, np.nan))[both]) / 2
        ranks = np.array([percentile(score[k]) for k in RULES])
        score["mix"] = ranks.mean(axis=0)  # NaN unless all four are known
        group = bottleneck.sectors(p.close, etf, mkt, t)
        for k in names:
            live = int(np.isfinite(score[k]).sum())
            eligible[k].append(live)
            enough = live >= cfg["min_eligible"]
            hold_until(f"{k}_top", t, top(score[k], cfg["top_share"]) if enough else everyone)
            hold_until(f"{k}_bottom", t, top(score[k], cfg["top_share"], highest=False) if enough else everyone)
            hold_until(f"{k}_sector", t, by_sector(score[k], group) if enough else everyone)
        for k in RULES:
            ok = np.isfinite(score[k]) & np.isfinite(gpa)
            corr[k].append(float(np.corrcoef(percentile(score[k][ok]), percentile(gpa[ok]))[0, 1]))
    ret = lambda key, c=cfg["cost_bps"]: run(p, x[key], x[key], v0, end, c, None, m.cash).returns
    h = simulate(p, exposures(p, *HOLD), v0, end, cfg["cost_bps"], m.cash).returns
    years = np.array([d.year for d in p.dates[v0:end]])
    half = np.arange(len(h)) < len(h) // 2
    sh = stats(h, rf)
    res: dict = dict(config="configs/filings_v23.yaml", window=f"{p.dates[v0]}..{p.dates[end - 1]}", hold=sh)
    for k in names:
        r = ret(f"{k}_top")
        s, bot, sect = stats(r, rf), stats(ret(f"{k}_bottom"), rf), stats(ret(f"{k}_sector"), rf)
        yearly = {int(y): dict(rule=round(float((1 + r[years == y]).prod() - 1) * 100, 1), hold=round(float((1 + h[years == y]).prod() - 1) * 100, 1)) for y in sorted(set(years))}
        best = sorted(yearly, key=lambda y: -(yearly[y]["rule"] - yearly[y]["hold"]))[:2]
        keep = ~np.isin(years, best)
        sharper = lambda i: stats(r[i], rf[i])["sharpe"] > stats(h[i], rf[i])["sharpe"]
        d = r - h
        edge, te = float(d.mean() * 252), float(d.std() * 252**0.5)
        tests = dict(test1=beats(s, sh), test2=bool(s["ann_return"] > bot["ann_return"]), test3=bool(sharper(keep)), test4=bool(sharper(half) and sharper(~half)), test5=beats(sect, sh))
        res[k] = dict(tests=tests, passed=all(tests.values()), rule=s, bottom_fifth=bot, by_sector=sect, at_5bps=stats(ret(f"{k}_top", cfg["side_cost_bps"]), rf), by_year=yearly, best_years=best,
                      edge_pct=round(edge * 100, 2), years_to_prove=round((2 * te / edge) ** 2, 1) if edge > 0 else None, eligible=[min(eligible[k]), max(eligible[k])],
                      monthly_turnover=round(float(np.abs(np.diff(x[f"{k}_top"][v0:end], axis=0)).sum(axis=1).mean() / n / 2 * 21), 3),
                      corr_with_quality=round(float(np.mean(corr[k])), 3) if k in corr else None)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"fetch": cmd_fetch, "run": cmd_run}[sys.argv[1]]()
