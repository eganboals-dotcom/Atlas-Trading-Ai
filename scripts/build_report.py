"""Charts for the write-up (docs/report/): built from this project's own results, never from raw market data.

Reads the walk-forward reports (artifacts/reports/swing/*/report.json), the controls summaries next to them,
the live paper sessions' daily reports (artifacts/paper/live_*/daily_report.json) and the recorded IEX quotes
(data/processed/alpaca_iex_live, used only for spread statistics). Writes PNGs plus figures/data.json, the
derived numbers behind every chart. Needs matplotlib: pip install -e .[report]

    .venv/Scripts/python.exe scripts/build_report.py
"""

from __future__ import annotations

import glob
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

OUT = Path("docs/report/figures")
SWING = Path("artifacts/reports/swing")

# Reference data-viz palette (light): one surface, recessive chrome, fixed categorical order.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, BAD = "#2a78d6", "#e34948"  # diverging poles: beat Hold / behind Hold
NEUTRAL = "#c3c2b7"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 12, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.labelcolor": INK2, "text.parse_math": False,
})


def titled(fig: plt.Figure, title: str, subtitle: str, source: str) -> None:
    fig.text(0.04, 0.955, title, fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.885, subtitle, fontsize=12.5, color=INK2, va="top")
    fig.text(0.04, 0.025, source, fontsize=9.5, color=MUTED, va="bottom")


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


def latest_reports() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(str(SWING / "*" / "report.json"))):  # run ids sort by time: the last one wins
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        out[r["config"]["experiment_name"]] = r
    return out


# 1. Backtests vs unseen days -------------------------------------------------------------------------------------
def overfitting(reports: dict[str, dict]) -> dict:
    rows = [{"experiment": n, "in_sample": r["selection"]["mean_in_sample_sharpe"],
             "unseen": r["selection"]["mean_oos_sharpe"], "trials": r["trials"]}
            for n, r in reports.items() if "selection" in r]
    x = np.array([r["in_sample"] for r in rows])
    y = np.array([r["unseen"] for r in rows])
    worse = int((y < x).sum())
    fig, ax = plt.subplots(figsize=(12, 7.2))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.14)
    lim = (-0.7, 3.8)
    ax.plot(lim, lim, color=MUTED, lw=1.2, ls=(0, (5, 4)))
    ax.text(3.0, 3.13, "if training scores\ntold the truth", color=MUTED, fontsize=10.5, ha="center", va="bottom",
            rotation=33)
    ax.scatter(x, y, s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    for r in rows:
        if r["in_sample"] > 3.0:
            ax.annotate("trained on only\n6 months", (r["in_sample"], r["unseen"]), xytext=(3.05, 1.55),
                        color=INK2, fontsize=10.5, ha="center",
                        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
            break
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xlim(*lim)
    ax.set_ylim(-0.7, 3.8)
    ax.set_xlabel("Best bot's Sharpe ratio in training")
    ax.set_ylabel("Same bot on days it had never seen")
    total = sum(r["trials"] for r in rows)
    titled(fig, f"Training scores overstated the result in {worse} of {len(rows)} experiments",
           f"Each dot is one experiment's best bot. Below the dashed line = it did worse on new days than in training."
           f"\n{total:,} bots tried in all.",
           "Source: Atlas walk-forward tests, 2016-2026 daily data; unseen = validation days after each training "
           "window. Research, not investment advice.")
    save(fig, "1_backtest_vs_unseen.png")
    return {"rows": rows, "worse": worse, "total_trials": total,
            "mean_gap": float((x - y).mean()), "corr": float(np.corrcoef(x, y)[0, 1])}


# 2. Live paper bots: trading more lost more ----------------------------------------------------------------------
def live_costs() -> dict:
    rows = []
    for f in sorted(glob.glob("artifacts/paper/live_*/daily_report.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        for a in r["accounts"]:
            if a["turnover"] > 0:
                rows.append({"day": r["session_date"], "bot": a["account"], "turnover": a["turnover"],
                             "return_bps": a["net_return"] * 1e4})
    fig, ax = plt.subplots(figsize=(12, 7.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.14)
    tx = np.array([r["turnover"] for r in rows])
    ry = np.array([r["return_bps"] for r in rows])
    hold = np.array([r["bot"] == "Hold" for r in rows])
    ax.axhline(0, color=AXIS, lw=1)
    ax.scatter(tx[~hold], ry[~hold], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3, label="trading bots")
    ax.scatter(tx[hold], ry[hold], s=90, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=4, marker="D",
               label="Hold (buy at the open, sell at the close)")
    days = sorted({r["day"] for r in rows})
    for r in rows:
        if r["bot"] in ("Random", "Flip", "Hold") or (r["bot"] == "Momentum" and r["day"] == days[-1]):
            right = r["turnover"] > 20
            ax.annotate(f"{r['bot']} ({r['day'][5:]})", (r["turnover"], r["return_bps"]),
                        xytext=(-8 if right else 8, 6), ha="right" if right else "left",
                        textcoords="offset points", color=INK2, fontsize=10)
    ax.set_xscale("log")
    ticks = [0.5, 1, 2, 5, 10, 20, 50]
    ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Turnover that day (value traded / capital, log scale)")
    ax.set_ylabel("Net return that day (bps; 100 bps = 1%)")
    ax.legend(loc="lower left")
    days = sorted({r["day"] for r in rows})
    corr = float(np.corrcoef(np.log(tx), ry)[0, 1])
    titled(fig, "On live prices, the bots that traded the most lost the most",
           f"Every bot on every live paper-trading day ({len(days)} so far): the more it traded, the more spread it paid.",
           "Source: Atlas live paper sessions on Alpaca IEX prices; orders simulated, none sent. Too few days to "
           "judge any bot; the point is the cost.")
    save(fig, "2_trading_costs_live.png")
    return {"rows": rows, "days": days, "corr_log_turnover_return": corr}


# 3. Live paper bots: before vs after costs ------------------------------------------------------------------------
def before_after_costs() -> dict:
    """Spread paid per fill = |fill price - quote midpoint| x shares, from the session audit logs."""
    cost: dict[str, float] = {}
    for f in sorted(glob.glob("artifacts/paper/live_*/audit.jsonl")):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e.get("type") == "execution" and e.get("status") == "filled" and e.get("fill_qty"):
                mid = (e["venue_bid"] + e["venue_ask"]) / 2
                cost[e["account"]] = cost.get(e["account"], 0.0) + abs(e["fill_price"] - mid) * e["fill_qty"]
    net: dict[str, float] = {}
    for f in sorted(glob.glob("artifacts/paper/live_*/daily_report.json")):
        for a in json.loads(Path(f).read_text(encoding="utf-8"))["accounts"]:
            net[a["account"]] = net.get(a["account"], 0.0) + a["net_pnl"]
    bots = [b for b in sorted(net, key=lambda b: net[b]) if cost.get(b, 0) > 0.01]
    gross = [net[b] + cost[b] for b in bots]
    fig, ax = plt.subplots(figsize=(12, 7.2))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.78, bottom=0.14)
    y = np.arange(len(bots))
    h = 0.36
    ax.barh(y + h / 2, gross, height=h, color=NEUTRAL, label="before trading costs")
    ax.barh(y - h / 2, [net[b] for b in bots], height=h, color=BLUE, label="after trading costs")
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, [f"{b} (the benchmark)" if b == "Hold" else b for b in bots])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Profit or loss in dollars, per $10,000 bot, over the live days")
    ax.legend(loc="lower right")
    paid = sum(cost[b] for b in bots if b != "Hold")
    lost = -sum(net[b] for b in bots if b != "Hold")
    titled(fig, "The trading bots' losses were almost exactly what they paid to trade",
           f"Live paper days so far: the trading bots paid ${paid:,.0f} crossing the spread and lost ${lost:,.0f} in all. "
           "Before costs,\nthey roughly broke even. Hold only bought at the open and sold at the close; it gained because the market rose.",
           "Source: Atlas live paper sessions on Alpaca IEX prices; every fill simulated at the quoted bid or ask, "
           "none sent. Spread cost = distance from the midpoint.")
    save(fig, "3_before_after_costs.png")
    return {"bots": bots, "net": [net[b] for b in bots], "cost": [cost[b] for b in bots], "gross": gross,
            "paid_trading_bots": paid, "lost_trading_bots": lost}


# 4. Scoreboard: every fixed rule on the 300 companies vs Hold ----------------------------------------------------
SHORT_NAMES = {  # report rule name -> chart label (plain words; "3 stocks" flags the concentrated ones)
    "dual momentum top 3 (12-1 months)": "Momentum: top 3 by 12-month return (3 stocks)",
    "sector momentum top 3 (6 months)": "Momentum: top 3 by 6-month return (3 stocks)",
    "value: top 10 by book-to-market (Fama-French), quarterly": "Value: top 10 by book-to-market",
    "quality: top 10 by gross profitability (Novy-Marx), quarterly": "Quality: top 10 by gross profitability",
    "earnings surprise (SUE): top 10, monthly": "Earnings surprise (filings): top 10",
    "earnings surprise (news, vs estimate): top 10, weekly": "Earnings surprise (headlines): top 10",
    "tilt: earnings surprise (news, vs estimate), tilt 0.5, weekly": "Tilt: earnings surprise (headlines)",
    "tilt: quality + earnings surprise + low short interest, tilt 0.5, monthly": "Tilt: quality + earnings + short (combined)",
    "tilt: low short interest, tilt 0.5, monthly": "Tilt: low short interest",
    "tilt: quality, tilt 0.5, quarterly": "Tilt: quality",
    "tilt: earnings surprise, tilt 0.5, monthly": "Tilt: earnings surprise (filings)",
    "tilt: insider buying clusters, tilt 0.5, monthly": "Tilt: insider buying",
    "tilt: momentum, tilt 0.5, quarterly": "Tilt: momentum",
    "tilt: quality + momentum + low volatility, tilt 0.5, quarterly": "Tilt: quality + momentum + low volatility",
    "quality + value (magic-formula style): top 10, quarterly": "Quality + value (magic formula): top 10",
    "dual momentum + 200-day trend + 15% vol cap": "Momentum + trend + volatility cap",
    "tilt: low volatility, tilt 0.5, quarterly": "Tilt: low volatility",
    "value: top 10 by earnings yield, quarterly": "Value: top 10 by earnings yield",
    "swing +5% target / -3% stop, up to 20 days": "Swing trade: +5% target / -3% stop",
    "200-day trend filter": "200-day trend filter",
    "volatility target 15%": "Volatility target 15%",
    "swing +10% target / -5% stop, above 200-day trend, up to 60 days": "Swing trade: +10% / -5%, above trend",
    "turn of the month (-1..+3)": "Turn of the month",
    "post-event drift (2% gap, 2x volume, 20 days)": "Post-gap drift (price only)",
    "insider clusters: hold companies with 2+ insiders buying (90 days)": "Insider buying clusters",
    "overnight only": "Overnight only",
    "day trade +1% target / -1% stop": "Day trade: +1% target / -1% stop",
}


def scoreboard(reports: dict[str, dict]) -> dict:
    rows: dict[str, dict] = {}
    hold = None
    for name in ("swing_v10_tilt_u300", "swing_v11_signals_u300", "swing_v12_news_eps_u300"):
        r = reports.get(name)
        if not r:
            continue
        hold = r["hold_base"]
        for k, v in r["others"].items():
            if k.startswith("rule:") and k[5:] in SHORT_NAMES:
                b = v["base"]
                rows[k[5:]] = {"label": SHORT_NAMES[k[5:]], "ann_return": b["ann_return"], "sharpe": b["sharpe"],
                               "excess": b["ann_return"] - hold["ann_return"], "passes": bool(r["passes"][k]),
                               "experiment": name}
        c = r["headline"]["champion"]["base"]["champion"]
        rows["champion"] = {"label": "Trained bots (best of thousands each period)", "ann_return": c["ann_return"],
                            "sharpe": c["sharpe"], "excess": c["ann_return"] - hold["ann_return"],
                            "passes": c["sharpe"] > hold["sharpe"] and c["ann_return"] >= hold["ann_return"],
                            "experiment": name}
    order = sorted(rows.values(), key=lambda r: r["excess"])
    fig, ax = plt.subplots(figsize=(12, 13))
    fig.subplots_adjust(left=0.40, right=0.94, top=0.88, bottom=0.07)
    y = np.arange(len(order))
    ex = np.array([r["excess"] * 100 for r in order])
    ax.barh(y, ex, height=0.62, color=[GOOD if v >= 0.05 else BAD if v <= -0.05 else NEUTRAL for v in ex])
    for i, r in enumerate(order):
        v = ex[i]
        ax.text(v + (0.4 if v >= 0 else -0.4), i, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right",
                color=INK2, fontsize=10)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, [r["label"] for r in order], fontsize=11)
    ax.tick_params(axis="y", colors=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(ex) - 4, max(ex) + 5)
    ax.set_xlabel("Yearly return minus Hold's (percentage points), unseen days, after 2 bps trading costs")
    beat = int((ex >= 0.05).sum())  # 0.0 after rounding is a tie, not a win
    fig.text(0.04, 0.975, f"Only {beat} of {len(order)} strategies earned more than simply holding all 300",
             fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.945, "The biggest winners held just 3 stocks at a time (the AI and electric-car run) and fell 54% at "
             "worst; most others land\nwithin a point or two of Hold, or well behind it. Hold: "
             f"{hold['ann_return']:+.1%} a year (equal weight).", fontsize=12.5, color=INK2, va="top")
    fig.text(0.04, 0.015, "Source: Atlas walk-forward tests on ~300 large US companies, unseen days 2019-2026. Fixed "
             "published rules, never tuned. Research, not investment advice.", fontsize=9.5, color=MUTED)
    save(fig, "4_scoreboard.png")
    return {"hold": hold, "rows": order, "beat": beat}


# 5. Best fifth vs worst fifth ------------------------------------------------------------------------------------
def fifths() -> dict:
    sig = json.loads((SWING / "signal_controls_u300.json").read_text(encoding="utf-8"))
    tc = json.loads((SWING / "tilt_controls_u300.json").read_text(encoding="utf-8"))
    hold = tc["hold"][0]
    groups = [("Quality (gross profitability)", tc["quality"]["top fifth"][0], tc["quality"]["bottom fifth"][0], hold),
              ("Earnings surprise (filings)", sig["earnings"]["top fifth (52)"][0], sig["earnings"]["bottom fifth (52)"][0],
               hold),
              ("Low short interest (from mid-2021)", sig["short"]["least-shorted fifth (59), from 2021-07"][0],
               sig["short"]["most-shorted fifth (59), from 2021-07"][0], sig["hold_from_2021_07"][0]),
              ("Momentum (12-month return)", tc["momentum"]["top fifth"][0], tc["momentum"]["bottom fifth"][0], hold),
              ("Low volatility", tc["low volatility"]["top fifth"][0], tc["low volatility"]["bottom fifth"][0], hold),
              ("Insider buying clusters (with vs without)", sig["insider"]["companies with a cluster"][0],
               sig["insider"]["companies without"][0], hold)]
    v12 = SWING / "v12_controls_u300.json"
    if v12.exists():
        d = json.loads(v12.read_text(encoding="utf-8"))["news_surprise"]
        top = next(v for k, v in d.items() if k.startswith("top fifth"))
        bot = next(v for k, v in d.items() if k.startswith("bottom fifth"))
        groups.insert(2, ("Earnings surprise (headlines)", top[0], bot[0], hold))
    rows = [{"signal": g, "best": b - h, "worst": w - h} for g, b, w, h in groups]
    fig, ax = plt.subplots(figsize=(12, 7.6))
    fig.subplots_adjust(left=0.29, right=0.95, top=0.78, bottom=0.13)
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.plot([r["worst"], r["best"]], [yi, yi], color=NEUTRAL, lw=3, solid_capstyle="round", zorder=2)
    ax.scatter([r["best"] for r in rows], y, s=110, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="the group the textbook says should do best")
    ax.scatter([r["worst"] for r in rows], y, s=110, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3,
               marker="D", label="the group it says should do worst")
    ax.axvline(0, color=INK2, lw=1)
    ax.text(0.15, len(rows) - 0.45, "Hold", color=INK2, fontsize=10.5)
    ax.set_yticks(y, [r["signal"] for r in rows], fontsize=11.5)
    ax.tick_params(axis="y", colors=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Yearly return minus Hold's (percentage points), unseen days, after costs")
    ax.legend(loc="lower right", fontsize=10.5)
    right = sum(r["best"] > r["worst"] for r in rows)
    titled(fig, f"Textbook signals sorted stocks the right way {right} times out of {len(rows)}",
           "Each signal splits the 300 companies into fifths. Blue right of orange = the 'good' stocks really did better.",
           "Source: Atlas controls on unseen days 2019-2026 (short interest from July 2021). Equal weight within "
           "each fifth. Research, not investment advice.")
    save(fig, "5_best_vs_worst_fifth.png")
    return {"rows": rows, "right_way": right}


# 6. How long it takes to prove a small edge ----------------------------------------------------------------------
def years_to_prove() -> dict | None:
    v12 = SWING / "v12_controls_u300.json"
    if not v12.exists():
        return None
    edges = json.loads(v12.read_text(encoding="utf-8"))["edges"]
    names = {"quality top 10, quarterly": "Quality: top 10", "filing surprise top 10, monthly":
             "Earnings surprise (filings): top 10", "low short interest tilt, from 2021-07": "Tilt: low short interest",
             "headline surprise top 10, weekly": "Earnings surprise (headlines): top 10",
             "headline surprise tilt, weekly": "Tilt: earnings surprise (headlines)",
             "combined tilt, monthly": "Tilt: quality + earnings + short"}
    rows = sorted(((names.get(k, k), v) for k, v in edges.items() if v["edge_pct"] > 0 and v["years_for_2se"]),
                  key=lambda kv: kv[1]["years_for_2se"])
    fig, ax = plt.subplots(figsize=(12, 7.2))
    fig.subplots_adjust(left=0.30, right=0.95, top=0.78, bottom=0.14)
    y = np.arange(len(rows))[::-1]
    yrs = [v["years_for_2se"] for _, v in rows]
    ax.barh(y, yrs, height=0.58, color=BLUE)
    for yi, (_, v) in zip(y, rows):
        ax.text(v["years_for_2se"] * 1.08, yi, f"{v['years_for_2se']:,.0f} years", va="center", color=INK2,
                fontsize=11, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    tested = max(v["years_tested"] for v in edges.values())
    for x, lab in ((tested, f"data we have ({tested:g} years)"), (40, "a 40-year career")):
        ax.axvline(x, color=MUTED, lw=1.2, ls=(0, (5, 4)))
        ax.text(x * 1.04, len(rows) - 0.45, lab, color=MUTED, fontsize=10)
    ax.set_xscale("log")
    xt = [1, 3, 10, 30, 100, 300, 1000, 3000]
    ax.set_xticks(xt, [f"{v:,}" for v in xt])
    ax.minorticks_off()
    ax.set_xlim(1, 8000)
    ax.set_yticks(y, [f"{n}\nedge {v['edge_pct']:+.1f}% a year, tracking error {v['tracking_error_pct']:.0f}%"
                      for n, v in rows], fontsize=11)
    ax.tick_params(axis="y", colors=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Years of data needed to be fairly sure the edge isn't luck (log scale)")
    titled(fig, "Proving that even our best rules aren't luck would take years to centuries",
           "Years until each rule's edge over Hold is two standard errors from zero: (2 x tracking error / edge) squared. "
           "Small\nedges that wander far from the market need far more data than anyone has.",
           "Source: Atlas unseen days 2019-2026 (short interest from July 2021); tracking error = yearly spread of the "
           "daily return difference vs Hold; edge = its average x 252 (a simple, not compounded, yearly figure).")
    save(fig, "6_years_to_prove.png")
    return {"edges": edges, "years_tested": tested}


def main() -> None:
    reports = latest_reports()
    data = {"built": datetime.now(UTC).isoformat(timespec="seconds"), "overfitting": overfitting(reports),
            "live_costs": live_costs(), "costs": before_after_costs(), "scoreboard": scoreboard(reports),
            "fifths": fifths(), "years_to_prove": years_to_prove()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in data.items()
                      if isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
