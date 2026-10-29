"""SEC EDGAR company financials and insider trades (read-only), one of Atlas's few network modules.

HTTP GET only, on public, free SEC files (SEC "EDGAR APIs" and "Insider Transactions Data Sets" pages):

    GET https://www.sec.gov/files/company_tickers.json                    ticker -> CIK
    GET https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json     every XBRL fact a company filed
    GET https://www.sec.gov/files/<structureddata|datastandardsinnovation>/data/insider-transactions-data-sets/
        YYYYqN_form345.zip                                                Forms 3, 4 and 5 for one quarter
        (added with the user's approval on 2026-10-28, for the insider-buying research)

The SEC requires a User-Agent naming who is asking, with a contact email ("Sample Company AdminContact@
sample.com"). It is read from ``SEC_USER_AGENT`` (environment or the git-ignored .env, via
``atlas.feeds.alpaca.read_env``); the user chooses what to put there. Requests are paced well under the
SEC's 10 per second.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from atlas.feeds.alpaca import read_env

UA_ENV = "SEC_USER_AGENT"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
INSIDER_URL = "https://www.sec.gov/files/{folder}/data/insider-transactions-data-sets/{year}q{quarter}_form345.zip"
INSIDER_FOLDERS = ("structureddata", "datastandardsinnovation")  # the SEC moved newer quarters to the second
_INSIDER_RE = re.compile(re.escape("https://www.sec.gov/files/") + r"(structureddata|datastandardsinnovation)/data/"
                         r"insider-transactions-data-sets/\d{4}q[1-4]_form345\.zip")
_FACTS_RE = re.compile(re.escape(FACTS_URL.split("CIK")[0]) + r"CIK\d{10}\.json")

Get = Callable[[str], dict[str, Any]]
GetFile = Callable[[str], bytes]


def _allowed(url: str, files: bool) -> bool:
    if files:
        return _INSIDER_RE.fullmatch(url) is not None
    return url == TICKERS_URL or _FACTS_RE.fullmatch(url) is not None


def user_agent(env_file: Path | None = Path(".env")) -> str:
    ua = read_env((UA_ENV,), env_file)[UA_ENV]
    if "@" not in ua:
        raise PermissionError(f"set {UA_ENV} in .env (yourself) to a name and contact email, e.g. "
                              f"'{UA_ENV}=Atlas research you@example.com'; the SEC requires one")
    return ua


def _downloader(ua: str, timeout_s: float, min_interval_s: float, files: bool) -> Callable[[str], bytes]:
    last = [0.0]

    def fetch(url: str) -> bytes:
        if not _allowed(url, files):
            raise PermissionError(f"refusing non-EDGAR url {url!r}")
        wait = last[0] + min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, method="GET", headers={
            "User-Agent": ua, "Accept": "application/zip" if files else "application/json"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    last[0] = time.monotonic()
                    return resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503) or attempt == 4:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == 4:
                    raise
            time.sleep(2.0 * (attempt + 1))
        raise RuntimeError("unreachable")

    return fetch


def http_get(ua: str, timeout_s: float = 60.0, min_interval_s: float = 0.25) -> Get:
    """JSON from the ticker file or a company's facts; nothing else."""
    fetch = _downloader(ua, timeout_s, min_interval_s, files=False)
    return lambda url: json.loads(fetch(url).decode("utf-8"))


def http_get_file(ua: str, timeout_s: float = 300.0, min_interval_s: float = 1.0) -> GetFile:
    """The bytes of one quarter's insider-transactions zip; nothing else."""
    return _downloader(ua, timeout_s, min_interval_s, files=True)


def insider_zip(get_file: GetFile, year: int, quarter: int) -> bytes | None:
    """One quarter's Forms 3/4/5 data set, from whichever SEC folder holds it; None if not published."""
    for folder in INSIDER_FOLDERS:
        try:
            return get_file(INSIDER_URL.format(folder=folder, year=year, quarter=quarter))
        except urllib.error.HTTPError as exc:
            if exc.code not in (403, 404):
                raise
    return None


# Earlier filers whose history a ticker's current EDGAR company lacks: in 2026 a new holding company took over
# XOM's ticker, and Exxon Mobil Corporation's filings before that stay under its own CIK; Disney's new parent
# (after the 2019 Fox deal) files from fiscal 2017 on, the old Walt Disney Co before that.
EARLIER_CIKS: dict[str, tuple[int, ...]] = {"XOM": (34088,), "DIS": (1001039,)}
# Companies no longer listed, so missing from EDGAR's ticker file (Walgreens went private in 2025).
DELISTED_CIKS: dict[str, int] = {"WBA": 1618921}


def ciks(get: Get) -> dict[str, int]:
    """Ticker -> CIK. EDGAR writes class shares with a dash (BRK-B); Alpaca with a dot (BRK.B)."""
    data = get(TICKERS_URL)
    return {row["ticker"].replace("-", "."): int(row["cik_str"]) for row in data.values()}


def company_facts(get: Get, cik: int) -> dict[str, Any]:
    return get(FACTS_URL.format(cik=cik))
