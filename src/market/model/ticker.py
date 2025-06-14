"""
Market ticker domain model.

This model represents ticker data in the domain layer, independent of any
specific exchange implementation. It implements MarketTickerProtocol through
its properties.

The model uses domain primitives internally for rich behavior while exposing
Decimal values via properties to satisfy the protocol's neutral type contract.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from ..domain.primitives import Percentage, Price, Size, Volume
from ..protocols.market import MarketTickerProtocol


class MarketTicker(BaseModel):
    """
    Market ticker with rich domain behavior.

    This model stores data as Decimal (satisfying the protocol) while
    providing rich domain methods that leverage domain primitives internally.
    This approach maintains simplicity while enabling rich behavior.
    """

    model_config = ConfigDict(
        populate_by_name=True, validate_assignment=True, str_strip_whitespace=True
    )

    # Core fields - stored as Decimal to satisfy protocol
    symbol: str = Field(..., min_length=1, max_length=20)
    exchange: str = Field(..., min_length=1, max_length=50)
    price: Decimal = Field(..., gt=0)
    size: Decimal = Field(..., gt=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Optional market data
    bid: Decimal | None = Field(None, gt=0)
    ask: Decimal | None = Field(None, gt=0)
    bid_size: Decimal | None = Field(None, gt=0)
    ask_size: Decimal | None = Field(None, gt=0)
    volume: Decimal | None = Field(None, ge=0)
    vwap: Decimal | None = Field(None, gt=0)

    # Computed domain primitives (cached for performance)
    @computed_field  # type: ignore[misc]
    @property
    def price_primitive(cls) -> Price:
        """Price as domain primitive for calculations."""
        return Price(value=cls.price)

    @computed_field  # type: ignore[misc]
    @property
    def size_primitive(cls) -> Size:
        """Size as domain primitive for calculations."""
        return Size(value=cls.size)

    @computed_field  # type: ignore[misc]
    @property
    def bid_primitive(cls) -> Price | None:
        """Bid price as domain primitive."""
        return Price(value=cls.bid) if cls.bid is not None else None

    @computed_field  # type: ignore[misc]
    @property
    def ask_primitive(cls) -> Price | None:
        """Ask price as domain primitive."""
        return Price(value=cls.ask) if cls.ask is not None else None

    @computed_field  # type: ignore[misc]
    @property
    def volume_primitive(cls) -> Volume | None:
        """Volume as domain primitive."""
        if cls.volume is not None:
            return Volume(size=Size(value=cls.volume))
        return None

    # Computed derived values
    @computed_field  # type: ignore[misc]
    @property
    def spread(cls) -> Decimal | None:
        """Calculate bid-ask spread."""
        if cls.bid is None or cls.ask is None:
            return None
        return cls.ask - cls.bid

    @computed_field  # type: ignore[misc]
    @property
    def mid_price(cls) -> Decimal | None:
        """Calculate mid price between bid and ask."""
        if cls.bid is None or cls.ask is None:
            return None
        return (cls.bid + cls.ask) / Decimal("2")

    # Rich domain operations
    def spread_percentage(self) -> Percentage | None:
        """Calculate spread as percentage of mid price."""
        mid = self.mid_price
        spread = self.spread

        if mid is None or spread is None:
            return None

        if mid == 0:
            return None

        percentage_value = (spread / mid) * 100
        return Percentage(value=percentage_value)

    def spread_basis_points(self) -> int | None:
        """Calculate spread in basis points."""
        pct = self.spread_percentage()
        if pct is None:
            return None

        bps = pct.as_basis_points()
        return int(bps.value)

    def price_change_from(self, other: MarketTickerProtocol) -> Percentage:
        """
        Calculate percentage change from another ticker.

        Note: 'other' is typed as protocol, not concrete model!
        This allows comparison with any protocol-compliant ticker.
        """
        other_price = Price(value=other.price)
        current_price = Price(value=self.price)
        return current_price.percentage_change(other_price)

    def is_more_liquid_than(self, other: MarketTickerProtocol) -> bool:
        """Compare liquidity with another ticker."""
        # Can compare with any protocol-compliant ticker
        if self.volume is None or other.volume is None:
            return False
        return self.volume > other.volume

    def is_inverted(self) -> bool:
        """Check if spread is inverted (bid > ask)."""
        if self.bid is None or self.ask is None:
            return False
        return self.bid > self.ask

    def is_liquid(
        self, min_volume: Decimal | None = None, max_spread_bps: int | None = None
    ) -> bool:
        """Check if market is liquid based on volume and spread criteria."""
        # Check volume if threshold provided
        if min_volume is not None and self.volume is not None:
            if self.volume < min_volume:
                return False

        # Check spread if threshold provided
        if max_spread_bps is not None:
            spread_bps = self.spread_basis_points()
            if spread_bps is None or spread_bps > max_spread_bps:
                return False

        return True

    def format_volume(self) -> str:
        """Format volume for display with proper units."""
        if self.volume is None:
            return "N/A"

        vol_value = self.volume
        if vol_value >= Decimal("1_000_000"):
            return f"{vol_value / Decimal('1_000_000'):.2f}M"
        elif vol_value >= Decimal("1_000"):
            return f"{vol_value / Decimal('1_000'):.2f}K"
        else:
            return f"{vol_value:.2f}"

    def format_summary(self) -> str:
        """Format a human-readable summary."""
        # Access computed properties works naturally
        parts = [
            f"{self.symbol} @ {self.exchange}",
            f"Price: {self.price_primitive.format_display()}",
        ]

        if self.bid and self.ask:
            parts.append(f"Bid/Ask: {self.bid}/{self.ask}")
            spread_bps = self.spread_basis_points()
            if spread_bps is not None:
                parts.append(f"Spread: {spread_bps}bps")

        vol_prim = self.volume_primitive
        if vol_prim:
            parts.append(f"Volume: {vol_prim.format_display()}")

        return " | ".join(parts)

    def is_stale(self, max_age_seconds: int = 300) -> bool:
        """Check if ticker data is stale based on timestamp."""
        now = datetime.now(UTC)
        age = now - self.timestamp
        return age.total_seconds() > max_age_seconds

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override to exclude computed fields by default."""
        # By default, exclude computed fields to keep serialization clean
        kwargs.setdefault("exclude", set()).update(
            {
                "price_primitive",
                "size_primitive",
                "bid_primitive",
                "ask_primitive",
                "volume_primitive",
                "spread",
                "mid_price",
            }
        )
        return super().model_dump(**kwargs)
