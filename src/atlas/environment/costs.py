"""Spread, slippage, and fee models. All adjustments are adverse to the trader."""

from __future__ import annotations

from atlas.config import FeeConfig, SlippageConfig, SpreadConfig
from atlas.environment.orders import Side


class CostModel:
    def __init__(self, fees: FeeConfig, slippage: SlippageConfig, spread: SpreadConfig) -> None:
        self.fees = fees
        self.slippage = slippage
        self.spread = spread

    def execution_price(self, side: Side, bid: float, ask: float) -> float:
        """Price paid/received: the far side of the quote, widened and slipped.

        Buys execute at the ask and sells at the bid — never at the midpoint.
        """
        adverse_bps = self.spread.widen_bps + self.slippage.bps
        if side is Side.BUY:
            return ask * (1.0 + adverse_bps / 10_000.0)
        return bid * (1.0 - adverse_bps / 10_000.0)

    def fee(self, side: Side, quantity: int, price: float) -> float:
        notional = quantity * price
        fee = self.fees.per_share * quantity + notional * self.fees.bps / 10_000.0
        fee = max(fee, self.fees.min_per_order)
        if side is Side.SELL:
            fee += notional * self.fees.sell_regulatory_bps / 10_000.0
        return fee
