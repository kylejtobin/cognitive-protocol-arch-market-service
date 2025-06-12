"""
Market snapshot model - simple Pydantic implementation of MarketSnapshotProtocol.

This model represents a point-in-time view of market state, implementing the
protocol through direct property mapping. It's immutable (frozen) to ensure
thread safety and functional programming patterns.
"""

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.market.protocols.market import (
    MarketOrderBookProtocol,
    MarketStatusProtocol,
    MarketTickerProtocol,
    MarketTradeProtocol,
)


class MarketSnapshot(BaseModel):
    """
    Simple market snapshot implementing the protocol.

    This is the core domain model that aggregates market state at a point in time.
    It's designed to be:
    - Immutable (frozen=True)
    - Protocol-compliant through direct property mapping
    - Simple with no complex logic
    """

    symbol: str
    timestamp: datetime
    ticker: MarketTickerProtocol | None = None
    order_book: MarketOrderBookProtocol | None = None
    trades: Sequence[MarketTradeProtocol] = Field(default_factory=list)
    status: MarketStatusProtocol | None = None

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @property
    def recent_trades(self) -> Sequence[MarketTradeProtocol] | None:
        """Alias for trades to match some protocol variations."""
        return self.trades if self.trades else None
