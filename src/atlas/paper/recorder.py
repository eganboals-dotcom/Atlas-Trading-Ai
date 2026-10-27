"""Record the live market data a paper session receives, to build a real-market dataset day by day.

Every event exactly as the feed delivered it (before validation, so the same rules can be
re-applied later) is appended to ``data/raw/<dataset>/<date>.csv``. At the end of the
session the day is also written as a normalized Parquet session into
``data/processed/<dataset>/`` and registered in that dataset's ``manifest.json``. The
existing ``ParquetProvider`` and walk-forward tools can then use it directly.

The recorder can never stop trading. Any error disables it with a warning, and the
session carries on. ``data/`` is git-ignored: recorded market data stays on this PC, for
the user's own research, and is never committed.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from atlas import __version__
from atlas.data.events import MarketEvent, Quote
from atlas.data.storage import SCHEMA, SCHEMA_VERSION, events_to_frame, frame_to_events, write_session

COLUMNS = list(SCHEMA)


class MarketDataRecorder:
    def __init__(self, root: Path, dataset: str, day: date, source: str, license_notes: str,
                 timestamp_kind: str = "vendor") -> None:
        self.dataset, self.day = dataset, day
        self.raw_path = root / "raw" / dataset / f"{day.isoformat()}.csv"
        self.processed_dir = root / "processed" / dataset
        self.source, self.license_notes, self.timestamp_kind = source, license_notes, timestamp_kind
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.raw_path.exists()
        self._fh = self.raw_path.open("a", newline="", encoding="utf-8")
        self._csv = csv.writer(self._fh)
        if new:
            self._csv.writerow(COLUMNS)
        self.rows = 0
        self.error: str | None = None

    def record(self, event: MarketEvent) -> None:
        if self.error:
            return
        row = events_to_frame([event]).row(0)
        self._csv.writerow(["" if v is None else v for v in row])
        self.rows += 1

    def flush(self) -> None:
        if not self.error:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def fail(self, exc: BaseException) -> None:
        self.error = f"{type(exc).__name__}: {exc}"
        try:
            self._fh.close()
        except OSError:
            pass

    def finalize(self, symbols: list[str]) -> dict[str, Any] | None:
        """Close the raw file and publish the day as a Parquet session in the dataset manifest."""
        if self.error:
            return None
        self._fh.close()
        events = read_raw_csv(self.raw_path)
        if not events:
            return None
        file_name = f"{self.day.isoformat()}.parquet"
        target = self.processed_dir / file_name
        sha = write_session(events, target)  # rewritten from the full raw file, so resumes are included
        manifest_path = self.processed_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {
            "schema_version": SCHEMA_VERSION, "dataset": self.dataset, "created_by": f"atlas {__version__}",
            "source": self.source, "feed": self.source, "license_notes": self.license_notes,
            "timestamp_kind": self.timestamp_kind,
            "timestamp_notes": "event_ts = Alpaca's RFC 3339 timestamp; receive_ts = this PC's clock on arrival",
            "fidelity": "Level 1 IEX quotes + trades, polled (latest quote ~1 s, latest trade ~2 s); not every tick",
            "price_adjustment": "none: raw intraday prices (sessions are simulated independently, flat at the open)",
            "corporate_actions": [], "universe_source": "symbols subscribed by the paper session",
            "chronology_notes": "events in local arrival order, exactly as the paper session received them",
            "raw_files": [], "sessions": [],
        }
        quotes = sum(isinstance(e, Quote) for e in events)
        entry = {"date": self.day.isoformat(), "file": file_name, "sha256": sha, "rows": len(events),
                 "quotes": quotes, "trades": len(events) - quotes, "symbols": sorted(symbols),
                 "recorded_at": datetime.now(UTC).isoformat()}
        manifest["sessions"] = sorted([s for s in manifest["sessions"] if s["date"] != entry["date"]] + [entry],
                                      key=lambda s: s["date"])
        manifest["raw_files"] = sorted({*manifest.get("raw_files", []), self.raw_path.as_posix()})
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp, manifest_path)
        return entry


def read_raw_csv(path: Path) -> list[MarketEvent]:
    """Parse a recorded raw CSV back into events (empty cells -> None)."""
    import polars as pl

    df = pl.read_csv(path, schema_overrides={c: SCHEMA[c] for c in COLUMNS}, infer_schema_length=0)
    df = df.with_columns(pl.col("conditions").fill_null(""))
    return frame_to_events(df)
