"""
Domain primitives for market data.

These primitives provide semantic meaning and rich behavior to values
while maintaining compatibility with protocol layer's neutral types.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    pass


class Price(BaseModel):
    """
    Represents a financial price with semantic operations.

    This primitive encapsulates price-related logic and formatting while
    providing conversion to Decimal for protocol satisfaction.
    """

    value: Decimal = Field(ge=Decimal("0"), description="Price value")
    currency: str = Field(default="USD", description="Currency code")

    model_config = ConfigDict(frozen=True)

    @field_validator("value")
    @classmethod
    def validate_price(cls, v: Decimal) -> Decimal:
        """Ensure price has reasonable precision."""
        # Limit to 8 decimal places (satoshi precision)
        return v.quantize(Decimal("0.00000001"))

    def to_decimal(self) -> Decimal:
        """Convert to Decimal for protocol satisfaction."""
        return self.value

    def as_float(self) -> float:
        """Convert to float (with precision loss)."""
        return float(self.value)

    def percentage_change(self, other: Price) -> Percentage:
        """Calculate percentage change from another price."""
        if other.value == 0:
            raise ValueError("Cannot calculate percentage change from zero price")
        if self.currency != other.currency:
            raise ValueError(f"Currency mismatch: {self.currency} vs {other.currency}")

        change = ((self.value - other.value) / other.value) * 100
        return Percentage(value=change)

    def with_percentage_change(self, pct: Percentage) -> Price:
        """Apply a percentage change to this price."""
        factor = 1 + pct.as_decimal()
        return Price(value=self.value * factor, currency=self.currency)

    def calculate_spread_to(self, other: Price) -> Spread:
        """Calculate spread to another price."""
        if self.currency != other.currency:
            raise ValueError(f"Currency mismatch: {self.currency} vs {other.currency}")

        return Spread(value=abs(self.value - other.value), reference_price=self)

    def format_display(self, decimals: int = 2) -> str:
        """Format price for display."""
        if self.currency == "USD":
            return f"${self.value:,.{decimals}f}"
        else:
            return f"{self.value:,.{decimals}f} {self.currency}"

    def __str__(self) -> str:
        """String representation."""
        return self.format_display()


class Size(BaseModel):
    """
    Represents a quantity/size of an asset.

    Encapsulates size-related operations and validations.
    """

    value: Decimal = Field(ge=Decimal("0"), description="Size/quantity value")

    model_config = ConfigDict(frozen=True)

    @field_validator("value")
    @classmethod
    def validate_size(cls, v: Decimal) -> Decimal:
        """Ensure size has reasonable precision."""
        # Limit to 8 decimal places
        return v.quantize(Decimal("0.00000001"))

    def to_decimal(self) -> Decimal:
        """Convert to Decimal for protocol satisfaction."""
        return self.value

    def as_float(self) -> float:
        """Convert to float (with precision loss)."""
        return float(self.value)

    def is_zero(self) -> bool:
        """Check if size is zero."""
        return self.value == 0

    def format_display(self, decimals: int = 4) -> str:
        """Format size for display."""
        return f"{self.value:,.{decimals}f}"

    def __add__(self, other: Size) -> Size:
        """Add two sizes."""
        return Size(value=self.value + other.value)

    def __sub__(self, other: Size) -> Size:
        """Subtract two sizes."""
        if other.value > self.value:
            raise ValueError("Cannot subtract larger size from smaller size")
        return Size(value=self.value - other.value)

    def __mul__(self, scalar: Decimal | int | float) -> Size:
        """Multiply size by scalar."""
        return Size(value=self.value * Decimal(str(scalar)))

    def __str__(self) -> str:
        """String representation."""
        return self.format_display()


class Percentage(BaseModel):
    """
    Represents a percentage value (50 = 50%).

    Provides conversions between percentage representations.
    """

    value: Decimal = Field(description="Percentage value (50 = 50%)")

    model_config = ConfigDict(frozen=True)

    def as_decimal(self) -> Decimal:
        """Convert to decimal (50% -> 0.5)."""
        return self.value / 100

    def as_basis_points(self) -> BasisPoints:
        """Convert to basis points (50% -> 5000bps)."""
        return BasisPoints(value=self.value * 100)

    def format_display(self, decimals: int = 2) -> str:
        """Format percentage for display."""
        return f"{self.value:.{decimals}f}%"

    def __str__(self) -> str:
        """String representation."""
        return self.format_display()


class BasisPoints(BaseModel):
    """
    Represents basis points (100 = 1%).

    Common in finance for expressing small percentages.
    """

    value: Decimal = Field(description="Basis points value (100 = 1%)")

    model_config = ConfigDict(frozen=True)

    def to_percentage(self) -> Percentage:
        """Convert to percentage (100bps -> 1%)."""
        return Percentage(value=self.value / 100)

    def as_decimal(self) -> Decimal:
        """Convert to decimal (100bps -> 0.01)."""
        return self.value / 10000

    def format_display(self, decimals: int = 1) -> str:
        """Format basis points for display."""
        return f"{self.value:.{decimals}f}bps"

    def __str__(self) -> str:
        """String representation."""
        return self.format_display()


class Spread(BaseModel):
    """
    Represents a price spread with reference to a base price.

    Provides multiple representations of the spread.
    """

    value: Decimal = Field(ge=Decimal("0"), description="Spread value")
    reference_price: Price = Field(
        description="Reference price for percentage calculations"
    )

    model_config = ConfigDict(frozen=True)

    def as_percentage(self) -> Percentage:
        """Express spread as percentage of reference price."""
        if self.reference_price.value == 0:
            return Percentage(value=Decimal("0"))
        pct = (self.value / self.reference_price.value) * 100
        return Percentage(value=pct)

    def as_basis_points(self) -> BasisPoints:
        """Express spread in basis points."""
        return self.as_percentage().as_basis_points()

    def is_tight(self, max_bps: Decimal = Decimal("10")) -> bool:
        """Check if spread is considered tight."""
        return self.as_basis_points().value <= max_bps

    def format_display(self) -> str:
        """Format spread for display."""
        return f"{self.reference_price.currency} {self.value:.4f} ({self.as_basis_points()})"

    def __str__(self) -> str:
        """String representation."""
        return self.format_display()


class Volume(BaseModel):
    """
    Represents trading volume over a time period.

    Encapsulates volume with temporal context.
    """

    size: Size = Field(description="Volume size")
    timeframe_hours: int = Field(default=24, gt=0, description="Timeframe in hours")

    model_config = ConfigDict(frozen=True)

    def daily_equivalent(self) -> Size:
        """Calculate 24-hour equivalent volume."""
        if self.timeframe_hours == 24:
            return self.size

        daily_multiplier = Decimal("24") / Decimal(str(self.timeframe_hours))
        return self.size * daily_multiplier

    def hourly_rate(self) -> Size:
        """Calculate hourly volume rate."""
        hourly_divisor = Decimal(str(self.timeframe_hours))
        return Size(value=self.size.value / hourly_divisor)

    def format_display(self) -> str:
        """Format volume for display."""
        if self.timeframe_hours == 24:
            return f"{self.size} (24h)"
        else:
            return f"{self.size} ({self.timeframe_hours}h)"

    def __str__(self) -> str:
        """String representation."""
        return self.format_display()


class VWAP(BaseModel):
    """
    Volume Weighted Average Price.

    Represents a price weighted by volume with confidence indicator.
    """

    price: Price = Field(description="The VWAP value")
    volume: Volume = Field(description="Total volume used in calculation")
    confidence: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Confidence score (0-1)",
    )

    model_config = ConfigDict(frozen=True)

    def to_price(self) -> Price:
        """Convert to Price for calculations."""
        return self.price

    def is_reliable(self, min_confidence: Decimal = Decimal("0.8")) -> bool:
        """Check if VWAP is reliable based on confidence."""
        return self.confidence >= min_confidence

    def format_display(self) -> str:
        """Format VWAP for display."""
        confidence_pct = self.confidence * 100
        return f"VWAP: {self.price} (conf: {confidence_pct:.0f}%)"

    def __str__(self) -> str:
        """String representation."""
        return self.format_display()

    @classmethod
    def calculate(cls, prices: list[Price], sizes: list[Size]) -> VWAP:
        """
        Calculate VWAP from price and size lists.

        Args:
            prices: List of prices
            sizes: List of corresponding sizes

        Returns:
            VWAP with calculated value and confidence

        """
        if not prices or not sizes:
            raise ValueError("Cannot calculate VWAP with empty data")

        if len(prices) != len(sizes):
            raise ValueError("Price and size lists must have same length")

        # Check currency consistency
        currencies = {p.currency for p in prices}
        if len(currencies) > 1:
            raise ValueError(f"Multiple currencies found: {currencies}")

        total_value = sum(
            p.value * s.value for p, s in zip(prices, sizes, strict=False)
        )
        total_size = sum(s.value for s in sizes)

        if total_size == 0:
            raise ValueError("Cannot calculate VWAP with zero total size")

        vwap_value = Decimal(str(total_value / total_size))
        total_volume = Volume(size=Size(value=Decimal(str(total_size))))

        # Simple confidence based on number of data points
        confidence = min(Decimal("1.0"), Decimal(str(len(prices))) / Decimal("100"))

        return cls(
            price=Price(value=vwap_value, currency=prices[0].currency),
            volume=total_volume,
            confidence=confidence,
        )
