"""Outer-holdout protection and the holdout access ledger.

Holdout sessions can only be turned into ``SessionRef`` objects through an unlocked
``HoldoutGuard``. Unlocking requires a *design fingerprint*: the hash of everything
that defines the candidate and the selection procedure. Every access is appended to a
ledger. A second access with a different design is refused unless the caller passes
``allow_reuse=True``; any reuse is recorded, and from then on reports state that the
holdout is no longer untouched.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from atlas.data.providers import MarketDataProvider, SessionRef


class HoldoutAccessError(PermissionError):
    pass


class HoldoutGuard:
    def __init__(self, provider: MarketDataProvider, holdout: tuple[date, ...]) -> None:
        self.provider = provider
        self.holdout = frozenset(holdout)

    def refs(self, sessions: list[date] | tuple[date, ...]) -> list[SessionRef]:
        """Refs for non-holdout sessions. Requesting any holdout session raises."""
        touched = sorted(set(sessions) & self.holdout)
        if touched:
            raise HoldoutAccessError(
                f"sessions {[d.isoformat() for d in touched]} are in the outer holdout; "
                "use `evaluate --holdout` with a frozen design"
            )
        return self.provider.refs(list(sessions))

    def unlock(self, ledger: HoldoutLedger, dataset_id: str, design_fingerprint: str, purpose: str,
               allow_reuse: bool = False) -> list[SessionRef]:
        ledger.authorize(dataset_id, design_fingerprint, sorted(self.holdout), purpose, allow_reuse)
        return self.provider.refs(sorted(self.holdout))


class HoldoutLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def accesses(self, dataset_id: str) -> list[dict[str, Any]]:
        return [e for e in self.entries() if e["dataset_id"] == dataset_id]

    def authorize(self, dataset_id: str, design: str, sessions: list[date], purpose: str, allow_reuse: bool) -> None:
        previous = self.accesses(dataset_id)
        other_designs = sorted({e["design_fingerprint"] for e in previous} - {design})
        if other_designs and not allow_reuse:
            raise HoldoutAccessError(
                f"the holdout for {dataset_id} was already used by design(s) {other_designs}; evaluating a new "
                "design on it would make it a validation set. Pass --allow-holdout-reuse to proceed anyway "
                "(this will be recorded and reported)."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "accessed_at": datetime.now(UTC).isoformat(),
            "dataset_id": dataset_id,
            "design_fingerprint": design,
            "sessions": [d.isoformat() for d in sessions],
            "purpose": purpose,
            "prior_accesses": len(previous),
            "reuse": bool(other_designs),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
