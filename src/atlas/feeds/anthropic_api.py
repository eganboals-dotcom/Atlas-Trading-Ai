"""Claude API client for the AI news reader (optional; the default reader runs a free model on this PC, see
``atlas.feeds.local_llm``). It is one of the few modules in Atlas allowed to use the network, and it makes one
kind of call: ``messages.create`` on Anthropic's API, with headline text in and a JSON list of notes out. It
has no market, order or account access of any kind.

The API key comes from ``ANTHROPIC_API_KEY`` (environment or the git-ignored .env), read through
``atlas.feeds.alpaca.read_env``. Install with ``pip install -e .[news]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atlas.feeds.alpaca import read_env
from atlas.news.prompt import CATEGORIES, NOTES_SCHEMA, SYSTEM, Classify  # noqa: F401  (CATEGORIES re-exported)

KEY_ENV = "ANTHROPIC_API_KEY"


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
