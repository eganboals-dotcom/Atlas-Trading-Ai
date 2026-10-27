"""Observation vectors for learned policies, plus a normalizer fit on training data only.

Per-symbol inputs (``SYMBOL_FEATURES``, normalized) followed by the symbol's position
value as a fraction of equity (raw, in [0, 1]). Global inputs: cash fraction and
session progress (both raw, in [0, 1]).

Missing values (e.g. returns before enough history exists) are encoded as 0 after
normalization together with explicit ``valid`` / ``history_ok`` flags, so the policy can
tell "unknown" from "zero".

The normalizer's statistics come only from the training sessions it was fit on. Its
fingerprint records those sessions, the schema version, and the statistics, and every
checkpoint stores it. A policy refuses to run with a mismatched schema.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from atlas.agents.features import FLOW_WINDOWS_S, RETURN_HORIZONS_S, FeatureEngine, SymbolFeatures
from atlas.data.providers import SessionRef, load_session
from atlas.environment.market_state import MarketView, PortfolioState
from atlas.timeutils import NS_PER_MS, ns_to_datetime, session_bounds_ns, MARKET_TZ

OBS_SCHEMA_VERSION = 1
SYMBOL_FEATURES: tuple[str, ...] = (
    "spread_bps", "l1_imbalance",
    *(f"ret_{h:g}s" for h in RETURN_HORIZONS_S),
    "vol_bps", "vol_ratio",
    *(f"flow_{w:g}s" for w in FLOW_WINDOWS_S),
    "log_volume_5s", "quote_age_ms", "valid", "history_ok",
)
N_SYMBOL_INPUTS = len(SYMBOL_FEATURES) + 1  # + position fraction
N_GLOBAL_INPUTS = 2  # cash fraction, session progress
_UNSCALED = {"valid", "history_ok"}
CLIP = 5.0


def raw_vector(f: SymbolFeatures) -> np.ndarray:
    def z(x: float) -> float:
        return 0.0 if x is None or not math.isfinite(x) else float(x)

    values = [
        z(f.spread_bps), z(f.l1_imbalance),
        *(z(f.returns_bps[h]) for h in RETURN_HORIZONS_S),
        z(f.vol_bps), z(f.vol_ratio),
        *(z(f.flow_imbalance[w]) for w in FLOW_WINDOWS_S),
        math.log1p(f.trade_volume[5.0]), z(f.quote_age_ms),
        1.0 if f.valid else 0.0,
        1.0 if math.isfinite(f.returns_bps[max(RETURN_HORIZONS_S)]) else 0.0,
    ]
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    fit_sessions: tuple[str, ...]
    samples: int
    schema_version: int = OBS_SCHEMA_VERSION

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    def transform(self, raw: np.ndarray) -> np.ndarray:
        out = (raw - np.asarray(self.mean)) / np.asarray(self.std)
        for i, name in enumerate(SYMBOL_FEATURES):
            if name in _UNSCALED:
                out[..., i] = raw[..., i]
        return np.clip(out, -CLIP, CLIP)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": list(self.mean), "std": list(self.std), "fit_sessions": list(self.fit_sessions),
                "samples": self.samples, "schema_version": self.schema_version,
                "features": list(SYMBOL_FEATURES)}

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> FeatureNormalizer:
        if d.get("schema_version") != OBS_SCHEMA_VERSION or list(d.get("features", [])) != list(SYMBOL_FEATURES):
            raise ValueError("normalizer was fit for a different observation schema")
        return FeatureNormalizer(tuple(d["mean"]), tuple(d["std"]), tuple(d["fit_sessions"]), int(d["samples"]))


def fit_normalizer(refs: list[SessionRef], decision_interval_ms: float, stale_ms: float) -> FeatureNormalizer:
    """Fit mean/std on the feature stream of the given (training) sessions only."""
    rows: list[np.ndarray] = []
    interval = int(decision_interval_ms * NS_PER_MS)
    for ref in refs:
        symbols = list(ref.symbols)
        view = MarketView(symbols, int(stale_ms * NS_PER_MS))
        engine = FeatureEngine(symbols, int(stale_ms * NS_PER_MS))
        next_t = None
        for e in sorted(load_session(ref), key=lambda ev: ev.receive_ts or ev.event_ts):
            t = e.receive_ts or e.event_ts
            view._advance(max(t, view.now_ns))
            if e.event_type == "quote":
                view._receive_quote(e)  # type: ignore[arg-type]
            else:
                view._receive_trade(e)  # type: ignore[arg-type]
            engine.update(view)
            if next_t is None or view.now_ns >= next_t:
                next_t = view.now_ns + interval
                rows.extend(raw_vector(f) for f in engine.snapshot(view.now_ns).values())
    if not rows:
        raise ValueError("no feature samples in training sessions")
    mat = np.vstack(rows)
    std = mat.std(axis=0)
    std[std < 1e-9] = 1.0
    return FeatureNormalizer(tuple(float(x) for x in mat.mean(axis=0)), tuple(float(x) for x in std),
                             tuple(r.session_date.isoformat() for r in refs), len(rows))


def session_progress(now_ns: int) -> float:
    day = ns_to_datetime(now_ns).astimezone(MARKET_TZ).date()
    start, end = session_bounds_ns(day)
    return min(1.0, max(0.0, (now_ns - start) / (end - start)))


def build_observation(
    snapshot: Mapping[str, SymbolFeatures], portfolio: PortfolioState, symbols: list[str],
    normalizer: FeatureNormalizer, now_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (per-symbol matrix [S, N_SYMBOL_INPUTS], global vector [N_GLOBAL_INPUTS]) as float32."""
    equity = max(portfolio.equity, 1e-9)
    rows = []
    for s in symbols:
        f = snapshot[s]
        mark = f.mid if f.valid and math.isfinite(f.mid) else 0.0
        pos_frac = portfolio.qty(s) * mark / equity
        rows.append(np.append(normalizer.transform(raw_vector(f)), min(1.0, max(0.0, pos_frac))))
    glob = np.asarray([min(1.0, max(0.0, portfolio.cash / equity)), session_progress(now_ns)])
    return np.asarray(rows, dtype=np.float32), glob.astype(np.float32)
