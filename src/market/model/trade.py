"""
Market trade domain model.

This model represents trade execution data in the domain layer, independent of any
specific exchange implementation. It implements MarketTradeProtocol through
its properties.
"""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from src.market.enums import TradeSide


class MarketTrade(BaseModel):
    """
    Domain model for market trade data.

    This is the canonical representation of trade data in the market service.
    Exchange-specific trade formats are transformed into this model at the
    adapter boundary.

    The model is frozen for immutability and thread safety.
    """

    symbol: str = Field(description="Market symbol (e.g., 'BTC-USD')")
    price: Decimal = Field(description="Executed trade price")
    size: Decimal = Field(description="Trade size in base currency")
    side: TradeSide = Field(description="Trade side (BUY or SELL)")
    timestamp: datetime = Field(description="Timestamp of trade execution")
    trade_id: str | None = Field(
        default=None, description="Unique trade identifier from exchange"
    )

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    @property
    def value(self) -> Decimal:
        """Calculate trade value (price * size)."""
        return self.price * self.size

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy trade."""
        return self.side == TradeSide.BUY

    @property
    def is_sell(self) -> bool:
        """Check if this is a sell trade."""
        return self.side == TradeSide.SELL
