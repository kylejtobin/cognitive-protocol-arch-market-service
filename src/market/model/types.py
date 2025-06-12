"""
Common types for market data models.

This module provides shared type definitions to ensure consistency
and type safety across the market data service.
"""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, field_validator


class PriceLevelData(BaseModel):
    """
    Represents a price level with price and size.

    Using Pydantic for consistency with the rest of the codebase
    and to get automatic validation of Decimal values.
    """

    price: Decimal
    size: Decimal

    model_config = ConfigDict(frozen=True)

    @field_validator("price", "size")
    @classmethod
    def validate_positive(cls, v: Decimal) -> Decimal:
        """Ensure price and size are positive."""
        if v < 0:
            raise ValueError("Price and size must be non-negative")
        return v

    def to_tuple(self) -> tuple[Decimal, Decimal]:
        """Convert to tuple for compatibility."""
        return (self.price, self.size)


# Since we're already using Pydantic everywhere, let's create
# proper models for other common types too
class PriceSize(BaseModel):
    """
    A simple price/size pair that satisfies MarketPriceLevelProtocol.

    This can be used when we need a protocol-compliant price level
    without the full PriceLevelSequence infrastructure.
    """

    price: Decimal
    size: Decimal

    model_config = ConfigDict(frozen=True)

    @field_validator("price", "size")
    @classmethod
    def validate_positive(cls, v: Decimal) -> Decimal:
        """Ensure price and size are positive."""
        if v < 0:
            raise ValueError("Price and size must be non-negative")
        return v
