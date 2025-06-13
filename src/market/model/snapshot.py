"""
Market snapshot model - simple Pydantic implementation.

This model represents a point-in-time view of market state. It uses
concrete Pydantic models for full type safety, validation, and serialization.
It's immutable (frozen) to ensure thread safety and functional programming patterns.
"""

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.market.model.book import OrderBook
from src.market.model.status import MarketStatus
from src.market.model.ticker import MarketTicker
from src.market.model.trade import MarketTrade


class MarketSnapshot(BaseModel):
    """
    Market snapshot with concrete models.

    This is the core domain model that aggregates market state at a point in time.
    It's designed to be:
    - Immutable (frozen=True)
    - Type-safe with concrete Pydantic models
    - Fully serializable
    - Simple with no complex logic
    """

    symbol: str
    timestamp: datetime
    ticker: MarketTicker | None = None
    order_book: OrderBook | None = None
    trades: Sequence[MarketTrade] = Field(default_factory=list)
    status: MarketStatus | None = None

    model_config = ConfigDict(frozen=True)

    @property
    def recent_trades(self) -> Sequence[MarketTrade] | None:
        """Alias for trades to match some protocol variations."""
        return self.trades if self.trades else None

    @property
    def has_complete_data(self) -> bool:
        """Check if snapshot has all major components."""
        return self.ticker is not None and self.order_book is not None

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        parts = [f"[{self.symbol}]"]

        if self.ticker:
            parts.append(f"Price: {self.ticker.price}")

        if self.order_book and self.order_book.spread:
            parts.append(f"Spread: {self.order_book.spread}")

        if self.status:
            parts.append(f"Status: {self.status.status.value}")

        return " ".join(parts)
