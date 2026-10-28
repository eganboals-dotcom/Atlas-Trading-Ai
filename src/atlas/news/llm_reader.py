"""Reader #2: an AI model reads each headline with company names hidden and writes notes.

Names are hidden for two reasons. The model shouldn't judge the company instead of the news ("distraction"),
and it shouldn't recognize a famous headline whose aftermath it remembers from training (look-ahead). For
the second reason the AI reader is only ever scored on headlines published after its start date, which is
after the model's training data ends: it is forward-tested, never backtested.

The network call is injected (``classify``), so this module never touches the network itself; the real one
is ``atlas.feeds.anthropic_api.claude_classifier``.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np
import polars as pl

from atlas.news.desk import NOTE_SCHEMA, empty_notes

# Names a headline might use for each tracked company (tickers are hidden too). Longest names first, so
# "Johnson & Johnson" is replaced before "Johnson".
ALIASES: dict[str, list[str]] = {
    "AAPL": ["Apple"], "GOOGL": ["Alphabet", "Google", "YouTube", "Waymo"], "GOOG": ["Alphabet", "Google"],
    "MSFT": ["Microsoft"], "BRK.B": ["Berkshire Hathaway", "Berkshire"], "XOM": ["Exxon Mobil", "ExxonMobil", "Exxon"],
    "AMZN": ["Amazon.com", "Amazon", "AWS"], "META": ["Meta Platforms", "Meta", "Facebook", "Instagram", "WhatsApp"],
    "GE": ["General Electric", "GE Aerospace", "GE"], "JNJ": ["Johnson & Johnson", "J&J"],
    "WFC": ["Wells Fargo"], "JPM": ["JPMorgan Chase", "JPMorgan", "JP Morgan"], "WMT": ["Walmart", "Wal-Mart"],
    "PG": ["Procter & Gamble", "P&G"], "VZ": ["Verizon Communications", "Verizon"], "T": ["AT&T"],
    "PFE": ["Pfizer"], "KO": ["Coca-Cola", "Coke"], "CVX": ["Chevron"], "ORCL": ["Oracle"],
    "DIS": ["Walt Disney", "Disney"], "MRK": ["Merck"], "PEP": ["PepsiCo", "Pepsi"], "HD": ["Home Depot"],
    "V": ["Visa"], "INTC": ["Intel"], "CSCO": ["Cisco Systems", "Cisco"], "BAC": ["Bank of America", "BofA"],
    "IBM": ["International Business Machines", "IBM"], "GILD": ["Gilead Sciences", "Gilead"],
    "CMCSA": ["Comcast", "NBCUniversal"], "MO": ["Altria"], "AMGN": ["Amgen"], "MA": ["Mastercard"],
    "C": ["Citigroup", "Citibank", "Citi"], "BA": ["Boeing"], "MMM": ["3M"],
    "UNH": ["UnitedHealth Group", "UnitedHealth", "Optum"], "MCD": ["McDonald's", "McDonalds"],
    "QCOM": ["Qualcomm"], "SLB": ["Schlumberger", "SLB"], "ABBV": ["AbbVie"], "CVS": ["CVS Health", "CVS", "Aetna"],
    "UPS": ["United Parcel Service", "UPS"], "BMY": ["Bristol-Myers Squibb", "Bristol Myers Squibb", "Bristol-Myers"],
    "MDT": ["Medtronic"], "NKE": ["Nike"], "SBUX": ["Starbucks"], "UNP": ["Union Pacific"],
    "HON": ["Honeywell"], "USB": ["U.S. Bancorp", "US Bancorp"], "LLY": ["Eli Lilly", "Lilly"],
    "TXN": ["Texas Instruments"], "BIIB": ["Biogen"], "MDLZ": ["Mondelez"], "COST": ["Costco Wholesale", "Costco"],
    "LOW": ["Lowe's", "Lowes"], "ACN": ["Accenture"], "ABT": ["Abbott Laboratories", "Abbott"],
    "GS": ["Goldman Sachs", "Goldman"],
}
LABELS = ("Company A", "Company B")
MAX_COMPANIES = len(LABELS)  # like the keyword reader, skip roundups that name many companies


def anonymize(headline: str, symbols: list[str]) -> str:
    """Replace each company's names and ticker with its label (Company A, Company B). One-letter tickers
    (T, C, V) are left alone: they would also hit words like the "T" in "T-Mobile"."""
    out = headline
    for label, sym in zip(LABELS, symbols):
        names = sorted(ALIASES.get(sym, []) + ([sym] if len(sym) > 1 else []), key=len, reverse=True)
        for name in names:
            out = re.sub(rf"(?<![\w&]){re.escape(name)}(?![\w&])(?:'s)?", label, out,
                         flags=0 if name.isupper() and len(name) <= 4 else re.IGNORECASE)
    return out


class LlmReader:
    """Reads articles about one or two tracked companies in batches through ``classify``."""

    def __init__(self, classify: Callable[[list[dict[str, Any]]], list[dict[str, Any]]], name: str = "claude",
                 batch_size: int = 25, tracked: set[str] | None = None) -> None:
        self.classify, self.name, self.batch_size = classify, name, batch_size
        self.tracked = tracked

    def read(self, articles: pl.DataFrame) -> pl.DataFrame:
        items, meta = [], []
        for aid, ns, syms, head in articles.select("id", "created_ns", "symbols", "headline").iter_rows():
            if not syms or len(syms) > MAX_COMPANIES or (self.tracked and not set(syms) & self.tracked):
                continue
            meta.append((aid, ns, list(syms)))
            items.append({"i": len(items), "headline": anonymize(head, list(syms)),
                          "companies": list(LABELS[:len(syms)])})
        rows = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start:start + self.batch_size]
            local = [{**it, "i": k} for k, it in enumerate(batch)]
            for n in self.classify(local):
                k = n.get("i")
                if not isinstance(k, int) or not 0 <= k < len(batch) or n.get("company") not in batch[k]["companies"]:
                    continue  # malformed or invented: ignore it rather than guess
                aid, ns, syms = meta[start + k]
                sym = syms[LABELS.index(n["company"])]
                rows.append((aid, ns, sym, self.name, float(np.clip(n["direction"], -1, 1)),
                             float(np.clip(n["importance"], 0, 1)), str(n["category"])))
        return pl.DataFrame(rows, schema=NOTE_SCHEMA, orient="row") if rows else empty_notes()
