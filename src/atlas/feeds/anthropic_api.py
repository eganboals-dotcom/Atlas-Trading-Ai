"""Claude API client for the AI news reader. Besides ``atlas.feeds.alpaca`` this is the only module in Atlas
allowed to use the network, and it makes one kind of call: ``messages.create`` on Anthropic's API, with
headline text in and a JSON list of notes out. It has no market, order or account access of any kind.

The API key comes from ``ANTHROPIC_API_KEY`` (environment or the git-ignored .env), read through
``atlas.feeds.alpaca.read_env``. Install with ``pip install -e .[news]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from atlas.feeds.alpaca import read_env

KEY_ENV = "ANTHROPIC_API_KEY"
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


def missing_key(env_file: Path | None = Path(".env")) -> bool:
    return not read_env((KEY_ENV,), env_file)[KEY_ENV]


def claude_classifier(model: str, env_file: Path | None = Path(".env"), max_tokens: int = 8000) -> Classify:
    """A function that sends one batch of anonymized headlines to Claude and returns its notes."""
    key = read_env((KEY_ENV,), env_file)[KEY_ENV]
    if not key:
        raise PermissionError(f"set {KEY_ENV} in .env (yourself) to use the AI news reader")
    import anthropic

    client = anthropic.Anthropic(api_key=key, max_retries=4, timeout=120.0)

    def classify(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": json.dumps({"headlines": items}, ensure_ascii=False)}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": NOTES_SCHEMA}},
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"Claude stopped with {response.stop_reason!r} (request {response._request_id})")
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)["notes"]

    return classify
