from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from helpers import data_config

from atlas.cli import EXIT_DATA_INVALID, EXIT_OK, main
from atlas.data.ingest import DataImportError, ImportConfig, export_synthetic_raw, import_dataset
from atlas.data.providers import ParquetProvider, SyntheticProvider, load_raw
from atlas.data.storage import events_to_frame, frame_to_events, read_session


def provider(n: int = 3) -> SyntheticProvider:
    return SyntheticProvider(data_config(), date(2026, 6, 1), n, duration_s=60, symbols=["SPY", "QQQ"])


def import_cfg(tmp: Path, **kw) -> ImportConfig:
    base = dict(dataset="demo", raw_dir=tmp / "raw", quotes_file="quotes.csv", trades_file="trades.csv",
                output_dir=tmp / "processed", source="test", timestamp_kind="synthetic", license_notes="test only",
                columns={"seq": "seq"})
    base.update(kw)
    return ImportConfig(**base)


def test_synthetic_provider_sessions_are_weekdays_with_distinct_seeds() -> None:
    p = provider(7)
    days = p.sessions()
    assert len(days) == 7 and all(d.weekday() < 5 for d in days)
    seeds = {p.session_config(d).synthetic.seed for d in days}
    assert len(seeds) == 7
    assert load_raw(p.session_ref(days[0])) == load_raw(p.session_ref(days[0]))


def test_frame_roundtrip_is_lossless() -> None:
    events = load_raw(provider().refs()[0])
    assert frame_to_events(events_to_frame(events)) == events


def test_export_import_roundtrip_matches_source(tmp_path: Path) -> None:
    src = provider()
    export_synthetic_raw(src, tmp_path / "raw")
    manifest = import_dataset(import_cfg(tmp_path, source=src.base.feed))
    assert [s["date"] for s in manifest["sessions"]] == [d.isoformat() for d in src.sessions()]
    assert all(len(s["sha256"]) == 64 for s in manifest["sessions"])
    assert manifest["raw_files"] and manifest["price_adjustment"].startswith("none")

    pq = ParquetProvider(tmp_path / "processed" / "demo")
    assert pq.sessions() == src.sessions()
    for d in src.sessions():
        assert load_raw(pq.session_ref(d)) == load_raw(src.session_ref(d))


def test_processed_data_is_immutable(tmp_path: Path) -> None:
    export_synthetic_raw(provider(1), tmp_path / "raw")
    import_dataset(import_cfg(tmp_path))
    with pytest.raises(DataImportError, match="immutable"):
        import_dataset(import_cfg(tmp_path))
    with pytest.raises(DataImportError, match="never overwritten"):
        export_synthetic_raw(provider(1), tmp_path / "raw")

    pq = ParquetProvider(tmp_path / "processed" / "demo")
    ref = pq.session_ref(pq.sessions()[0])
    df = pl.read_parquet(ref.path).with_columns(pl.col("bid") * 2)
    df.write_parquet(ref.path)
    with pytest.raises(ValueError, match="does not match manifest"):
        read_session(Path(ref.path), ref.sha256)
    assert main(["validate-data", "--dataset", str(tmp_path / "processed" / "demo")]) == EXIT_DATA_INVALID


def test_validate_dataset_cli_passes_clean_data(tmp_path: Path, capsys) -> None:
    export_synthetic_raw(provider(2), tmp_path / "raw")
    import_dataset(import_cfg(tmp_path))
    assert main(["validate-data", "--dataset", str(tmp_path / "processed" / "demo")]) == EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert len(out["sessions"]) == 2


def test_vendor_columns_iso_timestamps_universe_and_corporate_actions(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "q.csv").write_text(
        "ticker,time,b,a,bs,as\n"
        "spy,2026-06-01T13:30:00.000000001Z,100.00,100.02,500,600\n"
        "spy,2026-06-01T13:30:01Z,100.01,100.03,500,600\n"
        "spy,2026-06-02T13:30:00Z,101.00,101.02,500,600\n", encoding="utf-8")
    (raw / "t.csv").write_text("ticker,time,px,qty\nspy,2026-06-01T13:30:00.5Z,100.02,10\n", encoding="utf-8")
    (raw / "u.csv").write_text("date,symbol\n2026-06-01,SPY\n2026-06-01,XYZ\n2026-06-02,SPY\n", encoding="utf-8")
    (raw / "ca.csv").write_text("symbol,ex_date,type,ratio,amount\nSPY,2026-06-02,dividend,,1.5\n", encoding="utf-8")
    cfg = import_cfg(
        tmp_path, quotes_file="q.csv", trades_file="t.csv", timestamp_unit="iso", timestamp_kind="vendor",
        columns={"symbol": "ticker", "event_ts": "time", "bid": "b", "ask": "a", "bid_size": "bs",
                 "ask_size": "as", "price": "px", "size": "qty"},
        universe_file="u.csv", corporate_actions_file="ca.csv",
    )
    m = import_dataset(cfg)
    first, second = m["sessions"]
    assert first["rows"] == 3 and first["symbols"] == ["SPY", "XYZ"]
    assert first["missing_universe_symbols"] == ["XYZ"]
    assert second["corporate_actions"][0]["type"] == "dividend"
    events = load_raw(ParquetProvider(tmp_path / "processed" / "demo").session_ref(date(2026, 6, 1)))
    assert [e.event_type for e in events] == ["quote", "trade", "quote"]  # interleaved by source time
    assert events[0].symbol == "SPY" and events[0].event_ts % 10 == 1  # nanosecond precision kept


def test_import_requires_symbol_and_timestamp(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "q.csv").write_text("sym,bid\nSPY,1\n", encoding="utf-8")
    with pytest.raises(DataImportError, match="required column"):
        import_dataset(import_cfg(tmp_path, quotes_file="q.csv", trades_file=None))
