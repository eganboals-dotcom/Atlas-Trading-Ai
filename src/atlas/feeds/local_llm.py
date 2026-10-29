"""A free AI model on this PC for the news reader, served by Ollama (https://ollama.com). Approved by the user
on 2026-10-28 as Atlas's fourth network module, and the only one that never leaves the computer: it talks
to Ollama on the loopback address (127.0.0.1) only, with proxies and redirects turned off, so headlines are
never sent anywhere. It makes two kinds of request: list the downloaded models, and ask one of them to note a
batch of anonymized headlines. It has no market, order or account access of any kind, and needs no key.

Setup (the user, once): install Ollama, then ``ollama pull <model>`` for the model in configs/news_desk.yaml.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit

from atlas.news.prompt import NOTES_SCHEMA, SYSTEM, Classify

HOST = "http://127.0.0.1:11434"  # Ollama's default address
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
CHAT_PATH, TAGS_PATH = "/api/chat", "/api/tags"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # a redirect could point off this PC
        raise urllib.error.HTTPError(req.full_url, code, "redirects are refused", headers, fp)


# No proxy handler entries: a proxy set in the environment would otherwise carry headlines off this PC.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def check_host(host: str) -> str:
    """``host`` if it is plain http on this computer's loopback address; refuses anything else."""
    u = urlsplit(host)
    if u.scheme != "http" or u.hostname not in LOOPBACK or u.path not in ("", "/") or u.query or u.username:
        raise PermissionError(f"the local AI reader may only reach Ollama on this PC (http://127.0.0.1), not {host!r}")
    return host.rstrip("/")


def _request(host: str, path: str, body: dict[str, Any] | None, timeout_s: float) -> dict[str, Any]:
    if path not in (CHAT_PATH, TAGS_PATH):
        raise PermissionError(f"the local AI reader may not call {path!r}")
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(check_host(host) + path, data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if body is None else "POST")
    with _OPENER.open(req, timeout=timeout_s) as resp:
        return json.loads(resp.read())


def not_ready(model: str, host: str = HOST) -> str | None:
    """Why the local reader can't run right now (Ollama not running, model not downloaded), or None."""
    check_host(host)
    try:
        tags = _request(host, TAGS_PATH, None, timeout_s=5.0)
    except (OSError, ValueError):
        return "Ollama is not running on this PC (install it from ollama.com, then start it)"
    names = {m.get("name", "") for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        return f"model {model!r} is not downloaded (run: ollama pull {model})"
    return None


def local_classifier(model: str, host: str = HOST, timeout_s: float = 900.0, num_ctx: int = 8192) -> Classify:
    """A function that sends one batch of anonymized headlines to the local model and returns its notes."""
    check_host(host)

    def classify(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = _request(host, CHAT_PATH, {
            "model": model,
            "stream": False,
            "format": NOTES_SCHEMA,  # Ollama constrains the reply to this JSON schema
            "options": {"temperature": 0, "num_ctx": num_ctx},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": json.dumps({"headlines": items}, ensure_ascii=False)}],
        }, timeout_s)
        if out.get("done_reason") == "length":
            raise RuntimeError("the local model ran out of room; lower batch_size in configs/news_desk.yaml")
        return json.loads(out["message"]["content"])["notes"]

    return classify
