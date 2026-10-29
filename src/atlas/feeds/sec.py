"""SEC EDGAR company financials (read-only). The third and last module in Atlas allowed to use the network.

Two HTTP GET endpoints, both public and free (SEC "EDGAR APIs" documentation):

    GET https://www.sec.gov/files/company_tickers.json                    ticker -> CIK
    GET https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json     every XBRL fact a company filed

The SEC requires a User-Agent naming who is asking, with a contact email ("Sample Company AdminContact@
sample.com"). It is read from ``SEC_USER_AGENT`` (environment or the git-ignored .env, via
``atlas.feeds.alpaca.read_env``); the user chooses what to put there. Requests are paced well under the
SEC's 10 per second.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from atlas.feeds.alpaca import read_env

UA_ENV = "SEC_USER_AGENT"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

Get = Callable[[str], dict[str, Any]]


def user_agent(env_file: Path | None = Path(".env")) -> str:
    ua = read_env((UA_ENV,), env_file)[UA_ENV]
    if "@" not in ua:
        raise PermissionError(f"set {UA_ENV} in .env (yourself) to a name and contact email, e.g. "
                              f"'{UA_ENV}=Atlas research you@example.com'; the SEC requires one")
    return ua


def http_get(ua: str, timeout_s: float = 60.0, min_interval_s: float = 0.25) -> Get:
    last = [0.0]

    def get(url: str) -> dict[str, Any]:
        if not (url == TICKERS_URL or url.startswith("https://data.sec.gov/api/xbrl/companyfacts/CIK")):
            raise PermissionError(f"refusing non-EDGAR url {url!r}")
        wait = last[0] + min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": ua, "Accept": "application/json"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    last[0] = time.monotonic()
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503) or attempt == 4:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == 4:
                    raise
            time.sleep(2.0 * (attempt + 1))
        raise RuntimeError("unreachable")

    return get


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
