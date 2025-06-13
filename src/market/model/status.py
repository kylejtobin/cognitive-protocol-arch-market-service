"""
Market status domain model.

This model represents market operational status in the domain layer,
implementing MarketStatusProtocol through its properties.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.market.enums import MarketStatus as MarketStatusEnum


class MarketStatus(BaseModel):
    """
    Domain model for market operational status.

    This is the canonical representation of market status in the market service.
    It implements MarketStatusProtocol through direct property mapping.

    The model is frozen for immutability and thread safety.
    """

    symbol: str = Field(description="Market symbol (e.g., 'BTC-USD')")
    status: MarketStatusEnum = Field(description="Current market status")
    timestamp: datetime = Field(description="Timestamp of status update")
    reason: str | None = Field(default=None, description="Reason for status change")

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    def is_tradable(self) -> bool:
        """Check if market is currently tradable."""
        return self.status in {
            MarketStatusEnum.OPEN,
            MarketStatusEnum.LIMITED,
        }

    def to_log_entry(self) -> str:
        """Generate log-friendly representation."""
        reason_str = f" - {self.reason}" if self.reason else ""
        return f"[{self.symbol}] Status: {self.status.value}{reason_str}"
