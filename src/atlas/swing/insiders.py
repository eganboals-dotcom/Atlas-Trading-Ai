"""Insider buying from the SEC's quarterly Forms 3/4/5 data sets (``fetch-insider``), point in time.

``purchases`` keeps open-market purchases (transaction code P, shares acquired) by a company's directors and
officers, from Form 4 filings and their amendments; ten-percent holders that are neither (often funds) are left
out, as in the insider-trading research (Lakonishok and Lee, 2001; Cohen, Malloy and Pomorski, 2012). Each row
counts from the day after its filing date, when it was public. ``insider_features`` gives, for every day and
company, the number of different insiders who bought in the 90 calendar days before that day.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from atlas.swing.data import Panel

SCHEMA = {"issuer_cik": pl.Int64, "owner_cik": pl.Int64, "filed": pl.Date, "trans_date": pl.Date, "value": pl.Float64}
LOOKBACK_DAYS = 90


def _tsv(z: zipfile.ZipFile, name: str, cols: list[str]) -> pl.DataFrame:
    with z.open(name) as f:
        data = f.read()
    return pl.read_csv(io.BytesIO(data), separator="\t", quote_char=None, infer_schema=False, columns=cols,
                       encoding="utf8-lossy", truncate_ragged_lines=True)


def _day(s: str | None) -> date | None:
    try:
        return datetime.strptime(s, "%d-%b-%Y").date() if s else None
    except ValueError:
        return None


def purchases_from_zip(path: Path) -> pl.DataFrame:
    """One quarter's open-market purchases by directors and officers (the zip is only read, never run)."""
    with zipfile.ZipFile(path) as z:
        sub = _tsv(z, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK"])
        tr = _tsv(z, "NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                             "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
        own = _tsv(z, "REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
    sub = sub.filter(pl.col("DOCUMENT_TYPE").is_in(["4", "4/A"]))
    tr = tr.filter((pl.col("TRANS_CODE") == "P") & (pl.col("TRANS_ACQUIRED_DISP_CD") == "A"))
    rel = pl.col("RPTOWNER_RELATIONSHIP").fill_null("")
    own = own.filter(rel.str.contains("Director") | rel.str.contains("Officer"))
    df = tr.join(sub, on="ACCESSION_NUMBER").join(own, on="ACCESSION_NUMBER")
    if df.height == 0:
        return pl.DataFrame(schema=SCHEMA)
    num = lambda c: pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0)  # noqa: E731
    out = df.select(
        pl.col("ISSUERCIK").cast(pl.Int64, strict=False).alias("issuer_cik"),
        pl.col("RPTOWNERCIK").cast(pl.Int64, strict=False).alias("owner_cik"),
        pl.col("FILING_DATE").map_elements(_day, return_dtype=pl.Date).alias("filed"),
        pl.col("TRANS_DATE").map_elements(_day, return_dtype=pl.Date).alias("trans_date"),
        (num("TRANS_SHARES") * num("TRANS_PRICEPERSHARE")).alias("value"))
    return out.drop_nulls(["issuer_cik", "owner_cik", "filed"])


def build(zips_dir: Path, out: Path) -> pl.DataFrame:
    parts = [purchases_from_zip(f) for f in sorted(Path(zips_dir).glob("*_form345.zip"))]
    df = pl.concat(parts) if parts else pl.DataFrame(schema=SCHEMA)
    df = df.unique().sort("filed")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    return df


def insider_features(p: Panel, purchases: pl.DataFrame, ciks: dict[str, list[int]],
                     lookback_days: int = LOOKBACK_DAYS) -> dict[str, np.ndarray]:
    """``ins_buyers``: different directors and officers with open-market purchases filed in the
    ``lookback_days`` calendar days before each day (a filing counts from the next day). ``ciks``: each
    symbol's SEC filer numbers (a company that changed filer has several)."""
    out = np.zeros(p.close.shape)
    day_ord = np.array([d.toordinal() for d in p.dates])
    for j, sym in enumerate(p.symbols):
        rows = purchases.filter(pl.col("issuer_cik").is_in(ciks.get(sym, []))).select("owner_cik", "filed")
        if rows.height == 0:
            continue
        filed = np.array([d.toordinal() for d in rows["filed"].to_list()])
        owner = rows["owner_cik"].to_numpy()
        for i, t in enumerate(day_ord):
            m = (filed < t) & (filed >= t - lookback_days)
            if m.any():
                out[i, j] = len(set(owner[m].tolist()))
    return {"ins_buyers": out}


def load(insider_dir: Path) -> tuple[pl.DataFrame, dict[str, list[int]]]:
    """The purchases and the symbol -> SEC filer numbers written by ``build-insider``."""
    import json

    d = Path(insider_dir)
    return pl.read_parquet(d / "purchases.parquet"), json.loads((d / "ciks.json").read_text(encoding="utf-8"))
