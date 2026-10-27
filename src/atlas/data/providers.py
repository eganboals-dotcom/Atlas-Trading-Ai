"""Market data providers behind one interface.

A provider lists trading sessions and produces a ``SessionRef`` for each: a small,
serializable description that any process can turn back into the session's events.
Evaluation code only ever handles ``SessionRef`` objects, so evolution and walk-forward
work identically on synthetic and file-backed data.

Implemented: ``SyntheticProvider`` and ``ParquetProvider`` (normalized files written by
``atlas.data.ingest``). A live-data provider arrives with paper trading (Phase 5).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from atlas.config import DataConfig
from atlas.data.events import MarketEvent
from atlas.data.storage import read_session
from atlas.data.synthetic import generate_events
from atlas.data.validation import ValidationReport, validate_events


class SessionRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["synthetic", "parquet"]
    session_date: date
    symbols: tuple[str, ...]
    allowed_data_age_ms: float
    data_config: str | None = None  # synthetic: DataConfig JSON
    path: str | None = None  # parquet: session file
    sha256: str | None = None

    @property
    def key(self) -> str:
        return self.model_dump_json()


class MarketDataProvider(ABC):
    @abstractmethod
    def sessions(self) -> list[date]:
        """All available sessions, in chronological order."""

    @abstractmethod
    def session_ref(self, session: date) -> SessionRef: ...

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Source, license, timestamp provenance, fidelity, and corporate-action treatment."""

    def refs(self, sessions: list[date] | None = None) -> list[SessionRef]:
        return [self.session_ref(d) for d in (sessions if sessions is not None else self.sessions())]


def weekdays(start: date, count: int) -> list[date]:
    out, d = [], start
    while len(out) < count:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


class SyntheticProvider(MarketDataProvider):
    """Consecutive weekday sessions of synthetic data. Each date gets its own derived seed."""

    def __init__(self, base: DataConfig, start: date, n_sessions: int, duration_s: float | None = None,
                 symbols: list[str] | None = None) -> None:
        self.base = base
        self._sessions = weekdays(start, n_sessions)
        self.duration_s = duration_s or base.duration_s
        self.symbols = symbols or base.symbols

    def sessions(self) -> list[date]:
        return list(self._sessions)

    def session_config(self, session: date) -> DataConfig:
        seed = (self.base.synthetic.seed * 1_000_003 + session.toordinal()) % 2**32
        synthetic = self.base.synthetic.model_copy(update={"seed": seed})
        return DataConfig.model_validate({
            **self.base.model_dump(), "session_date": session, "duration_s": self.duration_s,
            "symbols": self.symbols, "synthetic": synthetic.model_dump(),
        })

    def session_ref(self, session: date) -> SessionRef:
        if session not in self._sessions:
            raise KeyError(f"no synthetic session {session}")
        cfg = self.session_config(session)
        return SessionRef(kind="synthetic", session_date=session, symbols=tuple(cfg.symbols),
                          allowed_data_age_ms=cfg.allowed_data_age_ms, data_config=cfg.model_dump_json())

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "synthetic",
            "source": self.base.feed,
            "license_notes": self.base.license_notes,
            "timestamp_kind": "synthetic",
            "fidelity": "Level 1 quotes + trades (synthetic random walk)",
            "price_adjustment": "not applicable (synthetic)",
            "corporate_actions": [],
            "universe": "fixed configured symbols",
            "sessions": len(self._sessions),
            "symbols": self.symbols,
        }


class ParquetProvider(MarketDataProvider):
    """Sessions from a normalized dataset directory containing ``manifest.json``."""

    def __init__(self, dataset_dir: str | Path, allowed_data_age_ms: float = 2_000.0) -> None:
        self.dir = Path(dataset_dir)
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no manifest.json in {self.dir}; run `atlas.cli import-data` first")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.allowed_data_age_ms = allowed_data_age_ms
        self._by_date = {date.fromisoformat(s["date"]): s for s in self.manifest["sessions"]}

    def sessions(self) -> list[date]:
        return sorted(self._by_date)

    def session_ref(self, session: date) -> SessionRef:
        s = self._by_date[session]
        return SessionRef(kind="parquet", session_date=session, symbols=tuple(s["symbols"]),
                          allowed_data_age_ms=self.allowed_data_age_ms,
                          path=str((self.dir / s["file"]).resolve()), sha256=s["sha256"])

    def describe(self) -> dict[str, Any]:
        m = self.manifest
        keys = ("dataset", "source", "feed", "license_notes", "timestamp_kind", "timestamp_notes", "fidelity",
                "price_adjustment", "corporate_actions", "universe_source", "chronology_notes")
        out = {k: m.get(k) for k in keys}
        out.update(provider="parquet", sessions=len(self._by_date))
        return out


def load_raw(ref: SessionRef) -> list[MarketEvent]:
    if ref.kind == "synthetic":
        assert ref.data_config is not None
        return generate_events(DataConfig.model_validate_json(ref.data_config))
    assert ref.path is not None
    return read_session(Path(ref.path), ref.sha256)


@lru_cache(maxsize=256)
def _load_validated(key: str) -> tuple[tuple[MarketEvent, ...], ValidationReport]:
    ref = SessionRef.model_validate_json(key)
    report = validate_events(load_raw(ref), ref.allowed_data_age_ms)
    return tuple(report.valid_events), report


def load_session(ref: SessionRef) -> list[MarketEvent]:
    """Validated events for a session; cached per process because data is immutable."""
    return list(_load_validated(ref.key)[0])


def session_validation(ref: SessionRef) -> ValidationReport:
    return _load_validated(ref.key)[1]


class DataSourceConfig(BaseModel):
    """Selects a provider from configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["synthetic", "parquet"]
    data_config: Path | None = None  # synthetic
    start_date: date | None = None  # synthetic
    n_sessions: int | None = None  # synthetic
    duration_s: float | None = None  # synthetic
    symbols: list[str] | None = None  # synthetic
    dataset_dir: Path | None = None  # parquet
    allowed_data_age_ms: float = 2_000.0  # parquet


def make_provider(cfg: DataSourceConfig, base_dir: Path | None = None) -> MarketDataProvider:
    from atlas.config import load_config

    def resolve(p: Path) -> Path:
        if p.exists() or base_dir is None:
            return p
        return base_dir / p

    if cfg.kind == "synthetic":
        if cfg.data_config is None or cfg.start_date is None or not cfg.n_sessions:
            raise ValueError("synthetic data_source needs data_config, start_date and n_sessions")
        base = load_config(resolve(cfg.data_config), DataConfig)
        return SyntheticProvider(base, cfg.start_date, cfg.n_sessions, cfg.duration_s, cfg.symbols)
    if cfg.dataset_dir is None:
        raise ValueError("parquet data_source needs dataset_dir")
    return ParquetProvider(resolve(cfg.dataset_dir), cfg.allowed_data_age_ms)
