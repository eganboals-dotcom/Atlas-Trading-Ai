"""FINRA equity short interest (read-only), one of Atlas's few network modules; added with the user's approval
on 2026-10-28 for the short-interest research. One kind of request, HTTP GET of a public file:

    GET https://cdn.finra.org/equity/otcmarket/biweekly/shrtYYYYMMDD.csv     short interest for one settlement date

FINRA publishes short interest twice a month (mid-month and month-end settlement dates), about a week and a half
after the settlement date. Files before June 2021 hold OTC securities only; from June 2021 they include
exchange-listed stocks. No key is needed.
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from typing import Callable

FILE_URL = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt{day:%Y%m%d}.csv"
_FILE_RE = re.compile(re.escape("https://cdn.finra.org/equity/otcmarket/biweekly/shrt") + r"\d{8}\.csv")
GetFile = Callable[[str], bytes]


def http_get_file(timeout_s: float = 120.0, min_interval_s: float = 0.5) -> GetFile:
    last = [0.0]

    def get(url: str) -> bytes:
        if _FILE_RE.fullmatch(url) is None:
            raise PermissionError(f"refusing non-short-interest url {url!r}")
        wait = last[0] + min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": "atlas-research/0.1",
                                                                 "Accept": "text/csv"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    last[0] = time.monotonic()
                    return resp.read()
            except urllib.error.HTTPError as exc:
                last[0] = time.monotonic()
                if exc.code not in (429, 500, 502, 503) or attempt == 3:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == 3:
                    raise
            time.sleep(2.0 * (attempt + 1))
        raise RuntimeError("unreachable")

    return get


def settlement_candidates(year: int, month: int) -> list[list[date]]:
    """Likely settlement dates for a month, each with fallbacks a few days earlier (weekends and holidays move
    them): the 15th, and the last day of the month."""
    mid = date(year, month, 15)
    end = (date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1))
    return [[mid - timedelta(days=k) for k in range(5)], [end - timedelta(days=k) for k in range(5)]]


def fetch_settlement(get: GetFile, candidates: list[date]) -> tuple[date, bytes] | None:
    for d in candidates:
        if d.weekday() >= 5:
            continue
        try:
            return d, get(FILE_URL.format(day=d))
        except urllib.error.HTTPError as exc:
            if exc.code not in (403, 404):
                raise
    return None
