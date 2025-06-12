"""
Market ticker domain model.

This model represents ticker data in the domain layer, independent of any
specific exchange implementation. It implements MarketTickerProtocol through
its properties.
"""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class MarketTicker(BaseModel):
    """
    Domain model for market ticker data.

    This is the canonical representation of ticker data in the market service.
    Exchange-specific ticker formats are transformed into this model at the
    adapter boundary.

    The model is frozen for immutability and thread safety.
    """

    symbol: str = Field(description="Market symbol (e.g., 'BTC-USD')")
    price: Decimal = Field(description="Latest traded price")
    bid: Decimal | None = Field(default=None, description="Best bid price")
    ask: Decimal | None = Field(default=None, description="Best ask price")
    volume: Decimal | None = Field(
        default=None, description="24-hour trading volume in base currency"
    )
    timestamp: datetime = Field(description="Timestamp of ticker update")

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    @property
    def spread(self) -> Decimal | None:
        """Calculate spread between bid and ask."""
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

    @property
    def mid_price(self) -> Decimal | None:
        """Calculate mid price between bid and ask."""
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2
