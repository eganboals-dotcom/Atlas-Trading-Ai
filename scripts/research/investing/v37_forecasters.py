"""Eight price forecasters on the 300 companies (see configs/forecasters_v37.yaml, which holds the pre-registered tests), on the unseen days.
run forecasts every week with each model (cached in artifacts/reports/swing/v37_forecasts/, so a stopped run resumes), then scores them and
writes artifacts/reports/swing/v37_forecasters.json. The foundation models' weights come from Hugging Face on first use.

    .venv/Scripts/python.exe scripts/research/investing/v37_forecasters.py run
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # TiRex-2 compiles a layer, and this PC has no C++ compiler; the config says so
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "trading"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v19_pairs as pairs  # noqa: E402
import v22_bottleneck as bottleneck  # noqa: E402
import v23_filings as v23  # noqa: E402
from atlas.swing.backtest import run, simulate  # noqa: E402
from atlas.swing.data import load_panel  # noqa: E402
from atlas.swing.families import HOLD, exposures  # noqa: E402

CONFIG = Path("configs/forecasters_v37.yaml")
cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
OUT = Path("artifacts/reports/swing/v37_forecasters.json")
CACHE = Path("artifacts/reports/swing/v37_forecasts")
H = cfg["horizon"]
LONGEST = 253  # sessions of history the features need


def rank(x: np.ndarray) -> np.ndarray:
    """Each row ranked across companies to 0..1."""
    return np.argsort(np.argsort(x, axis=-1, kind="stable"), axis=-1, kind="stable") / max(1, x.shape[-1] - 1)


def features(logp: np.ndarray, dollars: np.ndarray, t: int) -> np.ndarray:
    """(companies, 10) ranked features from rows before t, as the config lists them."""
    last = logp[t - 1]
    moves = [last - logp[t - 1 - k] for k in (1, 5, 21, 63, 126, 252)]
    daily = np.diff(logp[t - 253: t], axis=0)
    vols = [daily[-21:].std(axis=0), daily.std(axis=0)]
    high = [np.exp(last - logp[t - 252: t].max(axis=0))]
    flow = [dollars[t - 21: t].mean(axis=0) / dollars[t - 252: t].mean(axis=0)]
    return rank(np.array(moves + vols + high + flow)).T


def target(logp: np.ndarray, t: int) -> np.ndarray:
    """The next H sessions' log return from the close before t, ranked: known only from row t + H - 1 on."""
    return rank(logp[t + H - 1] - logp[t - 1])


def training_dates(t: int, first: int) -> list[int]:
    """Weekly dates on the rebalance grid in the train_years before t whose target ended before t."""
    lo = max(LONGEST, t - cfg["train_years"] * 252)
    return [s for s in range(first % cfg["every"], t, cfg["every"]) if s >= lo and s + H <= t]


def ar1(logp: np.ndarray, t: int) -> np.ndarray:
    """ARIMA(1,0,0) on each company's last 252 daily log returns, by least squares; the sum of the next H steps."""
    x = np.diff(logp[t - 253: t], axis=0)
    a, b = x[:-1], x[1:]
    phi = ((a - a.mean(0)) * (b - b.mean(0))).sum(0) / ((a - a.mean(0)) ** 2).sum(0)
    c = b.mean(0) - phi * a.mean(0)
    step, total = x[-1], np.zeros(x.shape[1])
    for _ in range(H):
        step = c + phi * step
        total += step
    return total


def sequences(logp: np.ndarray, t: int) -> np.ndarray:
    """(companies, window) daily log returns before t, each divided by its own standard deviation."""
    x = np.diff(logp[t - cfg["lstm"]["window"] - 1: t], axis=0).T
    return x / np.maximum(x.std(axis=1, keepdims=True), 1e-8)


class Trained:
    """The pooled models, refitted at the first rebalance of each year on the dates before it."""

    def __init__(self, kind: str, logp: np.ndarray, dollars: np.ndarray, first: int) -> None:
        self.kind, self.logp, self.dollars, self.first, self.model, self.year = kind, logp, dollars, first, None, None

    def _inputs(self, t: int) -> np.ndarray:
        return sequences(self.logp, t) if self.kind == "lstm" else features(self.logp, self.dollars, t)

    def _fit(self, t: int) -> None:
        dates = training_dates(t, self.first)
        x = np.concatenate([self._inputs(s) for s in dates])
        y = np.concatenate([target(self.logp, s) for s in dates])
        if self.kind == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(**cfg["random_forest"], random_state=cfg["seed"], n_jobs=-1).fit(x, y)
        elif self.kind == "xgboost":
            from xgboost import XGBRegressor
            self.model = XGBRegressor(**cfg["xgboost"], random_state=cfg["seed"], n_jobs=-1).fit(x, y)
        else:
            self.model = fit_lstm(x, y)

    def __call__(self, t: int, year: int) -> np.ndarray:
        if year != self.year:
            self._fit(t)
            self.year = year
        x = self._inputs(t)
        if self.kind == "lstm":
            import torch
            with torch.no_grad():
                return self.model(torch.tensor(x, dtype=torch.float32)).numpy()
        return self.model.predict(x)


def fit_lstm(x: np.ndarray, y: np.ndarray):
    import torch
    from torch import nn

    s = cfg["lstm"]
    torch.manual_seed(cfg["seed"])

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm, self.head = nn.LSTM(1, s["hidden"], batch_first=True), nn.Linear(s["hidden"], 1)

        def forward(self, seq: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(seq[:, :, None])
            return self.head(out[:, -1]).squeeze(-1)

    net, xs, ys = Net(), torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=s["learning_rate"])
    order = torch.Generator().manual_seed(cfg["seed"])
    for _ in range(s["epochs"]):
        for idx in torch.randperm(len(xs), generator=order).split(s["batch"]):
            opt.zero_grad()
            nn.functional.mse_loss(net(xs[idx]), ys[idx]).backward()
            opt.step()
    return net.eval()


def foundation(kind: str):
    """A function from (closes before t, all companies) to each company's forecast return over H sessions."""
    if kind == "timesfm":
        import timesfm
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(cfg["timesfm_repo"])
        m.compile(timesfm.ForecastConfig(max_context=cfg["context"], max_horizon=128, normalize_inputs=True, per_core_batch_size=64))
        return lambda closes: m.forecast(horizon=H, inputs=list(closes.T))[0][:, H - 1] / closes[-1] - 1
    import tirex2
    import torch
    m = tirex2.load_model(cfg["tirex_repo"], device="cpu", use_flex_attention=False)

    def tirex(closes: np.ndarray) -> np.ndarray:
        series = [tirex2.TimeseriesType(target=torch.tensor(c, dtype=torch.float32)[None], past_covariates=None, future_covariates=None) for c in closes.T]
        q = np.array([f[0] for f in m.forecast(series, prediction_length=H, output_type="numpy")])  # (companies, 9 quantiles, H)
        assert q.shape[1] == 9, q.shape  # 0.1 to 0.9; the middle one is the median
        return q[:, 4, H - 1] / closes[-1] - 1
    return tirex


def forecasts(kind: str, p, logp: np.ndarray, dollars: np.ndarray, dates: list[int]) -> np.ndarray:
    """(rebalances, companies) forecasts of one model, cached."""
    path = CACHE / f"{kind}.npy"
    if path.exists():
        return np.load(path)
    if kind in ("random_forest", "xgboost", "lstm"):
        model = Trained(kind, logp, dollars, dates[0])
        f = np.array([model(t, p.dates[t].year) for t in dates])
    elif kind in ("timesfm", "tirex"):
        model = foundation(kind)
        f = np.array([model(p.close[t - cfg["context"]: t]) for t in dates])
    else:
        one = dict(last_week=lambda t: logp[t - 1] - logp[t - 6], momentum=lambda t: logp[t - 22] - logp[t - 253], ar1=lambda t: ar1(logp, t))[kind]
        f = np.array([one(t) for t in dates])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, f)
    print(f"forecast {kind}: {f.shape}", flush=True)
    return f


def cmd_run() -> None:
    m = pairs.market(cfg["swing_config"])
    p, v0, end = m.p, m.v0, m.end
    n, days = p.close.shape[1], len(p.dates)
    assert v0 >= cfg["context"] and v0 >= LONGEST
    logp, dollars = np.log(p.close), p.close * p.volume
    dates = list(range(v0, end, cfg["every"]))
    rf = m.cash[v0:end]
    e = load_panel(Path(bottleneck.cfg["etf_dir"]), bottleneck.cfg["etfs"] + [bottleneck.cfg["market"]])
    at = {d: i for i, d in enumerate(e.dates)}
    idx = [at[d] for d in p.dates]
    etf, mkt = e.close[idx, :-1], e.close[idx, -1]
    spy = np.zeros(days)
    spy[1:] = mkt[1:] / mkt[:-1] - 1
    spy = spy[v0:end]
    groups = [bottleneck.sectors(p.close, etf, mkt, t) for t in dates]
    h = simulate(p, exposures(p, *HOLD), v0, end, cfg["cost_bps"], m.cash).returns
    years = np.array([d.year for d in p.dates[v0:end]])
    half = np.arange(len(h)) < len(h) // 2
    sh, ss = v23.stats(h, rf), v23.stats(spy, rf)
    res: dict = dict(config=str(CONFIG), config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(), window=f"{p.dates[v0]}..{p.dates[end - 1]}",
                     rebalances=len(dates), hold=sh, spy=ss)
    for kind in cfg["models"]:
        f = forecasts(kind, p, logp, dollars, dates)
        x = {side: np.zeros((days, n)) for side in ("top", "bottom", "sector")}
        for k, t in enumerate(dates):
            for side, picks in (("top", v23.top(f[k], cfg["top_share"])), ("bottom", v23.top(f[k], cfg["top_share"], highest=False)),
                                ("sector", v23.by_sector(f[k], groups[k]))):
                x[side][t: t + cfg["every"]] = 0.0
                x[side][t: t + cfg["every"], picks] = n / len(picks)
        ret = lambda side, c=cfg["cost_bps"]: run(p, x[side], x[side], v0, end, c, None, m.cash).returns  # noqa: E731
        r = ret("top")
        s, bot, sect = v23.stats(r, rf), v23.stats(ret("bottom"), rf), v23.stats(ret("sector"), rf)
        yearly = {int(y): {k: round(float((1 + a[years == y]).prod() - 1) * 100, 1) for k, a in (("rule", r), ("hold", h), ("spy", spy))} for y in sorted(set(years))}
        best = sorted(yearly, key=lambda y: -(yearly[y]["rule"] - yearly[y]["hold"]))[:2]
        keep = ~np.isin(years, best)
        sharper = lambda i: v23.stats(r[i], rf[i])["sharpe"] > v23.stats(h[i], rf[i])["sharpe"]  # noqa: E731
        ic = np.array([np.corrcoef(rank(f[k]), target(logp, t))[0, 1] for k, t in enumerate(dates) if t + H - 1 < end])
        tests = dict(test1=v23.beats(s, sh), test2=bool(s["ann_return"] > bot["ann_return"]), test3=bool(sharper(keep)),
                     test4=bool(sharper(half) and sharper(~half)), test5=v23.beats(sect, sh), test6=v23.beats(s, ss))
        res[kind] = dict(tests=tests, passed=all(tests.values()), rule=s, bottom_fifth=bot, by_sector=sect,
                         at_side_cost=v23.stats(ret("top", cfg["side_cost_bps"]), rf), by_year=yearly, best_years=best,
                         rank_correlation=dict(mean=round(float(ic.mean()), 4), t=round(float(ic.mean() / ic.std() * len(ic) ** 0.5), 2)),
                         weekly_turnover=round(float(np.abs(np.diff(x["top"][v0:end], axis=0)).sum(axis=1).sum() / n / 2 / len(dates)), 3))
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        print(f"{kind}: {sum(tests.values())}/6 {s['ann_return']:+.1%} Sharpe {s['sharpe']:.2f}, bottom {bot['ann_return']:+.1%}, IC {ic.mean():+.4f}", flush=True)
    print(f"Hold {sh['ann_return']:+.1%} ({sh['sharpe']:.2f}), SPY {ss['ann_return']:+.1%} ({ss['sharpe']:.2f})")


if __name__ == "__main__":
    {"run": cmd_run}[sys.argv[1]]()
