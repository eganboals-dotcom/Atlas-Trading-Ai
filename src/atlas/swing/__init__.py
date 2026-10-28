"""Multi-day (swing) research on daily bars: positions may be held overnight and for days.

Separate from the intraday event-replay stack. Every decision for day ``t`` is made at the open of ``t``
from information available then (closes up to ``t-1`` and the open of ``t``). It sets two exposures:
open-to-close of ``t``, and close of ``t`` to the open of ``t+1``. Research only: nothing here trades.
"""
