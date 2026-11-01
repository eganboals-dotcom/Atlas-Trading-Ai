"""The one holdout run of Hold plus the half-size Noise Area rule (see configs/noise_area_holdout_v18.yaml, which holds
the pre-registered tests). Reuses scripts/v15_noise_area.py unchanged. Writes artifacts/reports/swing/v18_holdout.json.

check scores days before the window and must reproduce scripts/v15_overlay.py. fetch downloads the locked months' bars to
their own folder, so nothing else can read them by accident. run logs itself in the holdout ledger first, and refuses to
run a second time.

    .venv/Scripts/python.exe scripts/v18_holdout.py check
    .venv/Scripts/python.exe scripts/v18_holdout.py fetch
    .venv/Scripts/python.exe scripts/v18_holdout.py run
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import yaml

import v15_noise_area as v

CONFIG = Path("configs/noise_area_holdout_v18.yaml")
cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
BARS = Path("data/processed/etf_1min_locked")
OUT = Path("artifacts/reports/swing/v18_holdout.json")
LEDGER = Path(cfg["ledger"])
COST = cfg["cost_bps"]


def score(first: str, last: str) -> dict:
    """Book, Hold and rule for each case on the days from first to last, on whichever bars v.BARS points at."""
    a, b = date.fromisoformat(first), date.fromisoformat(last)
    sims = {(s, d): v.simulate(s, d) for s in COST for d in (0, 1)}
    dates = sorted(x for x in set(sims[("SPY", 0)]["dates"]) & set(sims[("QQQ", 0)]["dates"]) if a <= x <= b)
    idx = {k: {d: i for i, d in enumerate(sim["dates"])} for k, sim in sims.items()}

    def aligned(key: tuple[str, int], x: np.ndarray) -> np.ndarray:
        return x[[idx[key][d] for d in dates]]

    legs = {(s, d): aligned((s, d), v.series(sims[(s, d)], c)) for s, c in COST.items() for d in (0, 1)}
    hold = np.mean([aligned((s, 0), v.hold_returns(s, sims[(s, 0)]["dates"])) for s in COST], axis=0)
    rule = {d: np.mean([legs[(s, d)] for s in COST], axis=0) for d in (0, 1)}
    book = {d: hold + cfg["half_size"] * rule[d] for d in (0, 1)}
    return dict(dates=dates, sims=sims, idx=idx, legs=legs, hold=hold, rule=rule, book=book)


def beats(a: dict, b: dict) -> bool:
    return a["sharpe"] > b["sharpe"] and a["ann_return"] >= b["ann_return"]


def flip_control(s: dict) -> dict:
    """The rule's Sharpe against the same trades and times with random directions."""
    rng = np.random.default_rng(cfg["seed"])
    rows = {k: [s["sims"][(k, 0)]["rows"][s["idx"][(k, 0)][d]] for d in s["dates"]] for k in COST}
    flips = []
    for _ in range(cfg["control_flips"]):
        total = np.zeros(len(s["dates"]))
        for k, c in COST.items():
            for i, row in enumerate(rows[k]):
                lg = [(p * (1.0 if rng.random() < 0.5 else -1.0), e, x) for p, e, x in row["base"]]
                total[i] += v.day_return(row["day"], lg, row["lev"], c / 1e4) / len(COST)
        flips.append(total.mean() / total.std() * v.YEAR**0.5 if total.std() else 0.0)
    return dict(beat_fraction=float((v.perf(s["rule"][0])["sharpe"] > np.array(flips)).mean()),
                flip_sharpe_95=float(np.percentile(flips, 95)), flip_sharpe_median=float(np.median(flips)))


def cmd_check() -> None:
    s = score(cfg["check_first"], cfg["check_last"])
    got = v.perf(s["book"][0])
    want = cfg["check_expected"]
    print("check", s["dates"][0], s["dates"][-1], round(got["ann_return"], 3), round(got["sharpe"], 2), "want", want)
    if abs(got["ann_return"] - want["ann_return"]) > 0.001 or abs(got["sharpe"] - want["sharpe"]) > 0.01:
        sys.exit("check failed: the overlay's figures were not reproduced")


def cmd_fetch() -> None:
    v.BARS = BARS
    stop = date.fromisoformat(cfg["locked_last"]) + timedelta(days=1)  # v.cmd_fetch stops the day before its holdout_start
    v.cfg.update(symbols=cfg["symbols"], first=cfg["warmup_first"], holdout_start=stop.isoformat())
    v.cmd_fetch()


def log_use() -> None:
    entries = [json.loads(x) for x in LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()]
    if any(e.get("rule") == cfg["rule"] for e in entries):
        sys.exit("refused: this rule has already used the locked months, see the ledger")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    entry = dict(evaluated_at=datetime.now(timezone.utc).isoformat(), rule=cfg["rule"], symbols=cfg["symbols"],
                 holdout=[cfg["locked_first"], cfg["locked_last"]], reuse_of_window=True, earlier_uses=len(entries),
                 config_sha256=sha(CONFIG), script_sha256=sha(__file__), rule_sha256=sha(v.__file__), rule_config_sha256=sha("configs/noise_area_v15.yaml"))
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def cmd_run() -> None:
    log_use()
    v.BARS = BARS
    s = score(cfg["locked_first"], cfg["locked_last"])
    hold, rule, book = v.perf(s["hold"]), v.perf(s["rule"][0]), v.perf(s["book"][0])
    late_rule, late = v.perf(s["rule"][1]), v.perf(s["book"][1])
    each = {k: v.perf(s["legs"][(k, 0)]) for k in COST}
    months = np.array([d.strftime("%Y-%m") for d in s["dates"]])
    tests = dict(test1=beats(book, hold), test2=beats(late, hold), test3=all(x["ann_return"] > 0 for x in each.values()))
    res = dict(config=str(CONFIG), first=str(s["dates"][0]), last=str(s["dates"][-1]), days=len(s["dates"]), tests=tests, passed=all(tests.values()),
               hold=hold, book=book, rule=rule, book_late=late, rule_late=late_rule, each=each, corr=float(np.corrcoef(s["rule"][0], s["hold"])[0, 1]),
               by_month={m: dict(hold=round(float((1 + s["hold"][months == m]).prod() - 1) * 100, 2), rule=round(float((1 + s["rule"][0][months == m]).prod() - 1) * 100, 2),
                                 book=round(float((1 + s["book"][0][months == m]).prod() - 1) * 100, 2)) for m in sorted(set(months))},
               period_return=dict(hold=float((1 + s["hold"]).prod() - 1), rule=float((1 + s["rule"][0]).prod() - 1), book=float((1 + s["book"][0]).prod() - 1), book_late=float((1 + s["book"][1]).prod() - 1)),
               control=flip_control(s), target=dict(mean_daily=book["mean_daily"], days_at_or_above_1pct=float((s["book"][0] >= 0.01).mean())))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(result_of=cfg["rule"], days=res["days"], tests=tests, passed=res["passed"], hold=hold, book=book, book_late=late)) + "\n")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"check": cmd_check, "fetch": cmd_fetch, "run": cmd_run}[sys.argv[1]]()
