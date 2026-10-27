"""Import raw vendor-style CSVs into the normalized per-session Parquet format.

Raw files are read, never modified. The output directory gets one Parquet file per
session plus ``manifest.json``, which records:
* source, license, and timestamp provenance;
* the SHA-256 of every raw and processed file;
* per-session row counts, symbols, and validation summaries;
* the point-in-time universe, the corporate actions, and how prices are treated.

Chronology assumption (also written to the manifest): quotes and trades usually arrive
in separate files, so they are interleaved by source timestamp. Ties keep quotes before
trades and preserve each file's row order. If the vendor supplies a cross-file sequence
number, map it to ``seq``; ``seq`` is then validated as well.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from atlas import __version__
from atlas.data.storage import SCHEMA, SCHEMA_VERSION, file_sha256, frame_to_events, write_session
from atlas.data.validation import validate_events

_UNIT_NS = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}


class ImportConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    raw_dir: Path
    quotes_file: str
    trades_file: str | None = None
    output_dir: Path = Path("data/processed")
    source: str
    feed: str = "unspecified"
    timezone: str = "America/New_York"
    timestamp_unit: Literal["ns", "us", "ms", "s", "iso"] = "ns"
    timestamp_kind: Literal["exchange", "vendor", "local", "synthetic"]
    timestamp_notes: str = ""
    license_notes: str
    fidelity: str = "Level 1 quotes + trades"
    columns: dict[str, str] = Field(default_factory=dict)  # normalized name -> raw column name
    universe_file: str | None = None  # CSV with columns: date, symbol
    corporate_actions_file: str | None = None  # CSV: symbol, ex_date, type, ratio, amount
    allowed_data_age_ms: float = 2_000.0


class DataImportError(RuntimeError):
    pass


def _read(path: Path, cfg: ImportConfig, event_type: str, order: int) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=10_000)
    rename = {raw: norm for norm, raw in cfg.columns.items() if raw in df.columns and raw != norm}
    df = df.rename(rename)
    for col in ("symbol", "event_ts"):
        if col not in df.columns:
            raise DataImportError(f"{path}: required column {col!r} missing (map it under `columns`)")
    df = df.with_columns(pl.lit(event_type).alias("event_type"), pl.lit(order).alias("_file_order"),
                         pl.int_range(pl.len()).alias("_row"))
    for ts_col in ("event_ts", "receive_ts"):
        if ts_col not in df.columns:
            continue
        if cfg.timestamp_unit == "iso":
            parsed = pl.col(ts_col).cast(pl.Utf8).str.to_datetime(time_zone="UTC", time_unit="ns")
            df = df.with_columns(parsed.dt.epoch("ns").alias(ts_col))
        else:
            df = df.with_columns((pl.col(ts_col).cast(pl.Int64) * _UNIT_NS[cfg.timestamp_unit]).alias(ts_col))
    return df


def _normalize(df: pl.DataFrame, cfg: ImportConfig) -> pl.DataFrame:
    defaults = {"source": cfg.source, "timestamp_kind": cfg.timestamp_kind, "conditions": ""}
    cols = []
    for name, dtype in SCHEMA.items():
        if name in df.columns:
            cols.append(pl.col(name).cast(dtype, strict=False))
        else:
            cols.append(pl.lit(defaults.get(name), dtype=dtype).alias(name))
    return df.select([*cols, "_file_order", "_row"]).with_columns(
        pl.col("symbol").str.strip_chars().str.to_uppercase()
    )


def import_dataset(cfg: ImportConfig) -> dict:
    raw_dir = cfg.raw_dir
    out_dir = cfg.output_dir / cfg.dataset
    if out_dir.exists() and any(out_dir.iterdir()):
        raise DataImportError(f"{out_dir} already exists; processed datasets are immutable — use a new dataset name")

    raw_files = [raw_dir / cfg.quotes_file] + ([raw_dir / cfg.trades_file] if cfg.trades_file else [])
    frames = [_read(raw_files[0], cfg, "quote", 0)]
    if cfg.trades_file:
        frames.append(_read(raw_files[1], cfg, "trade", 1))
    df = pl.concat([_normalize(f, cfg) for f in frames], how="vertical")
    df = df.sort(["event_ts", "_file_order", "_row"], maintain_order=True)
    df = df.with_columns(
        pl.from_epoch("event_ts", time_unit="ns").dt.replace_time_zone("UTC")
        .dt.convert_time_zone(cfg.timezone).dt.date().alias("_session")
    )

    universe: dict[date, list[str]] = {}
    if cfg.universe_file:
        u = pl.read_csv(raw_dir / cfg.universe_file, try_parse_dates=True)
        for d, sym in u.select("date", "symbol").iter_rows():
            universe.setdefault(d if isinstance(d, date) else date.fromisoformat(str(d)), []).append(sym.upper())
    corporate_actions = []
    if cfg.corporate_actions_file:
        corporate_actions = pl.read_csv(raw_dir / cfg.corporate_actions_file).to_dicts()

    sessions = []
    for (session,), part in sorted(df.group_by("_session", maintain_order=True), key=lambda kv: kv[0][0]):
        part = part.drop("_session", "_file_order", "_row")
        file_name = f"{session.isoformat()}.parquet"
        sha = write_session(part, out_dir / file_name)
        present = sorted(part["symbol"].unique().to_list())
        symbols = sorted(universe.get(session, present))
        report = validate_events(frame_to_events(part), cfg.allowed_data_age_ms)
        sessions.append({
            "date": session.isoformat(), "file": file_name, "sha256": sha, "rows": part.height,
            "symbols": symbols, "symbols_with_data": present,
            "missing_universe_symbols": sorted(set(symbols) - set(present)),
            "validation": report.summary(),
            "corporate_actions": [a for a in corporate_actions if str(a.get("ex_date")) == session.isoformat()],
        })

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": cfg.dataset,
        "created_by": f"atlas {__version__}",
        "source": cfg.source,
        "feed": cfg.feed,
        "license_notes": cfg.license_notes,
        "timestamp_kind": cfg.timestamp_kind,
        "timestamp_notes": cfg.timestamp_notes,
        "fidelity": cfg.fidelity,
        "timezone": cfg.timezone,
        "price_adjustment": (
            "none: raw unadjusted intraday prices. Each session is simulated independently from a flat, "
            "all-cash portfolio, so no position spans a split or dividend and no adjustment is applied "
            "(no double adjustment is possible). Multi-day features would require adjustment first."
        ),
        "corporate_actions": corporate_actions,
        "universe_source": "universe file" if cfg.universe_file else "derived from symbols present in each session",
        "chronology_notes": (
            "quotes and trades interleaved by source timestamp; ties keep quotes before trades and file row order"
            + ("" if "seq" in cfg.columns else "; no vendor sequence numbers supplied")
        ),
        "raw_files": [{"path": str(p.as_posix()), "sha256": file_sha256(p)} for p in raw_files],
        "sessions": sessions,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def export_synthetic_raw(provider, out_dir: Path) -> list[Path]:
    """Write a synthetic provider's sessions as raw vendor-style CSVs (quotes.csv, trades.csv)."""
    from atlas.data.providers import load_raw
    from atlas.data.storage import events_to_frame

    if out_dir.exists() and any(out_dir.iterdir()):
        raise DataImportError(f"{out_dir} already exists; raw data is never overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [events_to_frame(load_raw(ref)) for ref in provider.refs()]
    df = pl.concat(frames)
    quotes = df.filter(pl.col("event_type") == "quote").select(
        "symbol", "event_ts", "receive_ts", "seq", "bid", "ask", "bid_size", "ask_size")
    trades = df.filter(pl.col("event_type") == "trade").select(
        "symbol", "event_ts", "receive_ts", "seq", "price", "size")
    paths = [out_dir / "quotes.csv", out_dir / "trades.csv"]
    quotes.write_csv(paths[0])
    trades.write_csv(paths[1])
    return paths
