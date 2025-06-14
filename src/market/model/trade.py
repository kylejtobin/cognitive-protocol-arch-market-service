"""
Market trade domain model.

This model represents trade execution data in the domain layer, independent of any
specific exchange implementation. It implements MarketTradeProtocol through
its properties.
"""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from src.market.domain.primitives import Price, Size, Volume
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

    # Domain primitives as computed fields
    @computed_field  # type: ignore[prop-decorator]
    @property
    def price_primitive(cls) -> Price:
        """Get trade price as domain primitive."""
        return Price(value=cls.price)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def size_primitive(cls) -> Size:
        """Get trade size as domain primitive."""
        return Size(value=cls.size)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def volume_primitive(cls) -> Volume:
        """Get trade volume as domain primitive."""
        return Volume(size=cls.size_primitive)

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

    def price_impact_from(self, reference_price: Decimal) -> Decimal:
        """
        Calculate price impact from a reference price.

        Returns percentage difference from reference price.
        Uses Price primitive internally for rich calculation.
        """
        ref_price = Price(value=reference_price)
        return self.price_primitive.percentage_change(ref_price).value

    def is_large_trade(self, threshold: Decimal) -> bool:
        """Check if trade size exceeds threshold."""
        return self.size > threshold

    def format_summary(self) -> str:
        """Format trade as human-readable summary."""
        side_str = "BUY" if self.is_buy else "SELL"
        return (
            f"{side_str} {self.size_primitive.format_display()} @ "
            f"{self.price_primitive.format_display()}"
        )
