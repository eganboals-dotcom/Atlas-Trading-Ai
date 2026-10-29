"""What the AI news readers are asked, shared by every model they can use (Claude, or a local model)."""

from __future__ import annotations

from typing import Any, Callable

CATEGORIES = ("earnings", "guidance", "analyst", "product", "legal", "deal", "management", "macro", "other")

# One note per (headline, company) pair; the model returns them as a JSON object with a "notes" list.
NOTES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"notes": {"type": "array", "items": {
        "type": "object",
        "properties": {"i": {"type": "integer"}, "company": {"type": "string"},
                       "direction": {"type": "number"}, "importance": {"type": "number"},
                       "category": {"type": "string", "enum": list(CATEGORIES)}},
        "required": ["i", "company", "direction", "importance", "category"],
        "additionalProperties": False}}},
    "required": ["notes"],
    "additionalProperties": False,
}

SYSTEM = (
    "You read financial news headlines about large US companies. Company names are hidden: each company "
    "is written as Company A or Company B. For every headline and every company listed with it, judge how the "
    "news is likely to move that company's stock over the next few trading days.\n"
    "- direction: from -1 (clearly bad for the stock) to +1 (clearly good); 0 when neutral, mixed or unclear.\n"
    "- importance: from 0 (routine, no new information) to 1 (major: an earnings surprise, a guidance change, "
    "a large deal, a regulatory or legal decision).\n"
    "- category: the kind of news.\n"
    "Judge only from the headline text. Do not try to work out which company it is. Return exactly one note "
    "for each (headline, company) pair you are given, using the headline's number as i."
)

Classify = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
