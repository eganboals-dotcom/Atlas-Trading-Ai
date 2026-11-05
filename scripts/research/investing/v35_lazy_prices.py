"""Companies whose annual report barely changed (see configs/lazy_prices_v35.yaml, which holds the pre-registered tests), on the
unseen days. fetch stores each 10-K's words from SEC EDGAR; check prints how many were found, and no return; run writes
artifacts/reports/swing/v35_lazy_prices.json.

    .venv/Scripts/python.exe scripts/research/investing/v35_lazy_prices.py fetch
    .venv/Scripts/python.exe scripts/research/investing/v35_lazy_prices.py check
    .venv/Scripts/python.exe scripts/research/investing/v35_lazy_prices.py run
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import sys
import urllib.error
from collections import Counter
from datetime import date
from html import unescape
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "trading"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v19_pairs as pairs  # noqa: E402
import v22_bottleneck as bottleneck  # noqa: E402
import v23_filings as v23  # noqa: E402
from atlas.config import load_config  # noqa: E402
from atlas.feeds import sec  # noqa: E402
from atlas.swing.backtest import run, simulate  # noqa: E402
from atlas.swing.data import load_panel  # noqa: E402
from atlas.swing.families import HOLD, exposures  # noqa: E402
from atlas.swing.walkforward import SwingConfig  # noqa: E402

CONFIG = Path("configs/lazy_prices_v35.yaml")
cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
OUT = Path("artifacts/reports/swing/v35_lazy_prices.json")
TEXT = Path(cfg["text_dir"])
FIRST, LAST = date.fromisoformat(cfg["first_filed"]), date.fromisoformat(cfg["last_scored"])
DROP = re.compile(r"<ix:header\b.*?</ix:header>|<script\b.*?</script>|<style\b.*?</style>", re.S | re.I)
WORD = re.compile(r"[a-z]{%d,}" % cfg["min_letters"])


def words(html: str) -> str:
    """The document's words, lowercased, as the config says."""
    return " ".join(WORD.findall(unescape(re.sub(r"<[^>]+>", " ", DROP.sub(" ", html))).lower()))


def annual_reports(facts: dict) -> dict[str, str]:
    """Accession number -> filing date of every 10-K the company files cite, filed in the window the config reads."""
    out: dict[str, str] = {}
    for taxonomy in facts.get("facts", {}).values():
        for item in taxonomy.values():
            for unit in item.get("units", {}).values():
                for f in unit:
                    if f.get("form") in cfg["forms"] and f.get("accn") and FIRST <= date.fromisoformat(f["filed"]) < LAST:
                        out[f["accn"]] = min(f["filed"], out.get(f["accn"], f["filed"]))
    return out


def cmd_fetch() -> None:
    symbols = load_config(Path(cfg["swing_config"]), SwingConfig).symbols
    ua = sec.user_agent(Path(".env"))
    get, get_page = sec.http_get(ua), sec.http_get_page(ua)
    ciks = sec.ciks(get)
    for i, sym in enumerate(symbols):
        folder, cik = TEXT / sym, ciks.get(sym, sec.DELISTED_CIKS.get(sym))
        index = folder / "filings.json"
        if index.exists() or cik is None:
            continue
        folder.mkdir(parents=True, exist_ok=True)
        found = {}
        for c in (*sec.EARLIER_CIKS.get(sym, ()), cik):
            for accn, filed in annual_reports(sec.company_facts(get, c)).items():
                try:
                    doc = sec.ten_k_document(get_page, c, accn)
                    if doc:
                        html = get_page(f"{sec.ARCHIVE}{c}/{accn.replace('-', '')}/{doc}").decode("utf-8", "replace")
                        (folder / f"{accn}.txt.gz").write_bytes(gzip.compress(words(html).encode("utf-8")))
                except urllib.error.HTTPError as exc:  # a filing EDGAR no longer serves: recorded as missing, not fatal
                    doc = f"missing ({exc.code})"
                found[accn] = dict(filed=filed, cik=c, doc=doc)
        index.write_text(json.dumps(found, indent=1, sort_keys=True), encoding="utf-8")  # written last, so a stopped run redoes the company
        print(f"{i + 1}/{len(symbols)} {sym}: {sum(1 for f in found.values() if f['doc'] and not f['doc'].startswith('missing'))} 10-Ks", flush=True)


def filings(sym: str) -> list[tuple[date, Path]]:
    """(filed, text) of each stored 10-K, oldest first."""
    index = TEXT / sym / "filings.json"
    if not index.exists():
        return []
    rows = json.loads(index.read_text(encoding="utf-8"))
    return sorted((date.fromisoformat(r["filed"]), TEXT / sym / f"{a}.txt.gz") for a, r in rows.items() if (TEXT / sym / f"{a}.txt.gz").exists())


def cosine(a: Counter, b: Counter) -> float:
    dot = sum(n * b[w] for w, n in a.items() if w in b)
    norm = math.sqrt(sum(n * n for n in a.values()) * sum(n * n for n in b.values()))
    return dot / norm if norm else math.nan


def pair(rows: list[tuple[date, Path]], d: date) -> tuple[Path, Path] | None:
    """The latest 10-K filed before day d and the one filed prior_days before it, as the config says."""
    before = [r for r in rows if r[0] < d]
    if not before or (d - before[-1][0]).days > cfg["max_age_days"]:
        return None
    lo, hi = cfg["prior_days"]
    prior = [r for r in before[:-1] if lo <= (before[-1][0] - r[0]).days <= hi]
    return (before[-1][1], min(prior, key=lambda r: abs((before[-1][0] - r[0]).days - 365))[1]) if prior else None


class Scores:
    """Cosine similarity of a company's latest 10-K and the year before's, each pair read once."""

    def __init__(self) -> None:
        self._counts: dict[Path, Counter] = {}
        self._sim: dict[tuple[Path, Path], float] = {}

    def counts(self, path: Path) -> Counter:
        if path not in self._counts:
            self._counts[path] = Counter(gzip.decompress(path.read_bytes()).decode("utf-8").split())
        return self._counts[path]

    def __call__(self, rows: list[tuple[date, Path]], d: date) -> float:
        p = pair(rows, d)
        if p is None:
            return math.nan
        if p not in self._sim:
            self._sim[p] = cosine(self.counts(p[0]), self.counts(p[1]))
        return self._sim[p]


def cmd_check() -> None:
    """What was stored, and how many companies each rebalance can score. No return is computed."""
    symbols = load_config(Path(cfg["swing_config"]), SwingConfig).symbols
    rows = {s: filings(s) for s in symbols}
    sizes = sorted(len(gzip.decompress(p.read_bytes()).split()) for r in rows.values() for _, p in r)
    print(f"companies with 10-Ks: {sum(1 for r in rows.values() if r)} of {len(symbols)}; 10-Ks: {len(sizes)}")
    print("words per 10-K: smallest", sizes[:5], "median", sizes[len(sizes) // 2], "largest", sizes[-3:])
    for d in (date(2019, 1, 24), date(2021, 6, 1), date(2023, 6, 1), date(2025, 12, 1)):
        print(d, "eligible:", sum(1 for r in rows.values() if pair(r, d)))
    print("missing:", [s for s, r in rows.items() if not r])


def cmd_run() -> None:
    m = pairs.market(cfg["swing_config"])
    p, v0, end = m.p, m.v0, m.end
    n, days = p.close.shape[1], len(p.dates)
    rf = m.cash[v0:end]
    e = load_panel(Path(bottleneck.cfg["etf_dir"]), bottleneck.cfg["etfs"] + [bottleneck.cfg["market"]])
    at = {d: i for i, d in enumerate(e.dates)}
    idx = [at[d] for d in p.dates]
    etf, mkt = e.close[idx, :-1], e.close[idx, -1]
    spy = np.zeros(days)
    spy[1:] = mkt[1:] / mkt[:-1] - 1
    spy = spy[v0:end]
    rows = {s: filings(s) for s in p.symbols}
    similarity = Scores()
    x = {side: np.zeros((days, n)) for side in ("top", "bottom", "sector")}
    eligible: list[int] = []
    everyone = list(range(n))

    def hold_until(key: str, t: int, picks: list[int]) -> None:
        x[key][t: t + cfg["every"]] = 0.0
        x[key][t: t + cfg["every"], picks] = n / len(picks)

    for t in range(v0, end, cfg["every"]):
        score = np.array([similarity(rows[s], p.dates[t]) for s in p.symbols])
        live = int(np.isfinite(score).sum())
        eligible.append(live)
        enough = live >= cfg["min_eligible"]
        hold_until("top", t, v23.top(score, cfg["top_share"]) if enough else everyone)
        hold_until("bottom", t, v23.top(score, cfg["top_share"], highest=False) if enough else everyone)
        hold_until("sector", t, v23.by_sector(score, bottleneck.sectors(p.close, etf, mkt, t)) if enough else everyone)
    ret = lambda key, c=cfg["cost_bps"]: run(p, x[key], x[key], v0, end, c, None, m.cash).returns  # noqa: E731
    h = simulate(p, exposures(p, *HOLD), v0, end, cfg["cost_bps"], m.cash).returns
    r = ret("top")
    years = np.array([d.year for d in p.dates[v0:end]])
    half = np.arange(len(h)) < len(h) // 2
    s, sh, ss = v23.stats(r, rf), v23.stats(h, rf), v23.stats(spy, rf)
    bot, sect = v23.stats(ret("bottom"), rf), v23.stats(ret("sector"), rf)
    yearly = {int(y): {k: round(float((1 + a[years == y]).prod() - 1) * 100, 1) for k, a in (("rule", r), ("hold", h), ("spy", spy))} for y in sorted(set(years))}
    best = sorted(yearly, key=lambda y: -(yearly[y]["rule"] - yearly[y]["hold"]))[:2]
    keep = ~np.isin(years, best)
    sharper = lambda i: v23.stats(r[i], rf[i])["sharpe"] > v23.stats(h[i], rf[i])["sharpe"]  # noqa: E731
    gap = r - h
    edge, te = float(gap.mean() * 252), float(gap.std() * 252**0.5)
    tests = dict(test1=v23.beats(s, sh), test2=bool(s["ann_return"] > bot["ann_return"]), test3=bool(sharper(keep)),
                 test4=bool(sharper(half) and sharper(~half)), test5=v23.beats(sect, sh), test6=v23.beats(s, ss))
    res = dict(config=str(CONFIG), config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(), window=f"{p.dates[v0]}..{p.dates[end - 1]}",
               tests=tests, passed=all(tests.values()), rule=s, hold=sh, spy=ss, bottom_fifth=bot, by_sector=sect,
               at_5bps=v23.stats(ret("top", cfg["side_cost_bps"]), rf), by_year=yearly, best_years=best, eligible=[min(eligible), max(eligible)],
               edge_pct=round(edge * 100, 2), years_to_prove=round((2 * te / edge) ** 2, 1) if edge > 0 else None,
               monthly_turnover=round(float(np.abs(np.diff(x["top"][v0:end], axis=0)).sum(axis=1).mean() / n / 2 * 21), 3))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"fetch": cmd_fetch, "check": cmd_check, "run": cmd_run}[sys.argv[1]]()
