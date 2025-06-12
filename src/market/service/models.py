"""
Pydantic models for the technical analysis service layer.

These models provide type-safe request/response structures and
ensure data validation at service boundaries.
"""

from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.market.analysis.base import BaseIndicator


class PriceSeries(BaseModel):
    """
    Validated wrapper for price series data.

    Ensures price series meet requirements for indicator calculations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbol: str = Field(description="Trading symbol")
    series: pd.Series = Field(description="Time-indexed price series")
    last_updated: datetime = Field(description="Last update timestamp")

    @field_validator("series")
    @classmethod
    def validate_series(cls, v: pd.Series) -> pd.Series:
        """Ensure series has datetime index and numeric values."""
        if not isinstance(v.index, pd.DatetimeIndex):
            raise ValueError("Price series must have datetime index")

        if not pd.api.types.is_numeric_dtype(v):
            raise ValueError("Price series must contain numeric values")

        if v.isna().any():
            raise ValueError("Price series cannot contain NaN values")

        return v

    @property
    def length(self) -> int:
        """Get the number of price points."""
        return len(self.series)

    @property
    def latest_price(self) -> float | None:
        """Get the most recent price."""
        return float(self.series.iloc[-1]) if len(self.series) > 0 else None

    def has_minimum_data(self, periods: int) -> bool:
        """Check if series has minimum required data points."""
        return len(self.series) >= periods

    def to_list(self) -> list[float]:
        """Convert to list of floats."""
        return [float(x) for x in self.series]


class CacheStats(BaseModel):
    """Cache statistics with validation."""

    enabled: bool = Field(description="Whether caching is enabled")
    size: int = Field(default=0, ge=0, description="Current number of cached items")
    max_size: int = Field(default=0, ge=0, description="Maximum cache size")
    ttl: int = Field(default=0, ge=0, description="Cache TTL in seconds")
    hit_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Cache hit rate")

    @property
    def utilization(self) -> float:
        """Calculate cache utilization percentage."""
        return (self.size / self.max_size * 100) if self.max_size > 0 else 0.0

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        if not self.enabled:
            return "Cache disabled"
        return (
            f"Cache: {self.size}/{self.max_size} items "
            f"({self.utilization:.1f}% full), "
            f"TTL: {self.ttl}s, Hit rate: {self.hit_rate:.1%}"
        )


class IndicatorRequest(BaseModel):
    """Request to calculate indicators."""

    symbol: str = Field(description="Trading symbol")
    indicators: list[str] | None = Field(
        default=None,
        description="Specific indicators to calculate (None = all enabled)",
    )
    timestamp: datetime = Field(description="Calculation timestamp")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        return v.upper()


class IndicatorResult(BaseModel):
    """Result of indicator calculation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbol: str = Field(description="Trading symbol")
    timestamp: datetime = Field(description="Calculation timestamp")
    indicators: list[BaseIndicator] = Field(
        default_factory=list, description="Calculated indicators"
    )
    errors: list[str] = Field(
        default_factory=list, description="Any errors during calculation"
    )

    @property
    def success(self) -> bool:
        """Check if calculation was successful."""
        return len(self.errors) == 0 and len(self.indicators) > 0

    @property
    def indicator_names(self) -> list[str]:
        """Get names of calculated indicators."""
        return [type(ind).__name__ for ind in self.indicators]

    def get_by_type(self, indicator_type: type[BaseIndicator]) -> BaseIndicator | None:
        """Get indicator by type."""
        for indicator in self.indicators:
            if isinstance(indicator, indicator_type):
                return indicator
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "indicators": [ind.model_dump() for ind in self.indicators],
            "errors": self.errors,
            "success": self.success,
        }


class BatchCalculationRequest(BaseModel):
    """Request for batch indicator calculation."""

    requests: list[IndicatorRequest] = Field(
        description="Individual calculation requests"
    )
    batch_size: int = Field(
        default=10, ge=1, le=100, description="Max concurrent calculations per batch"
    )

    @property
    def symbols(self) -> list[str]:
        """Get unique symbols in batch."""
        return list({req.symbol for req in self.requests})

    @property
    def size(self) -> int:
        """Get total number of requests."""
        return len(self.requests)


class BatchCalculationResult(BaseModel):
    """Result of batch indicator calculation."""

    results: list[IndicatorResult] = Field(description="Individual calculation results")
    total_time_ms: float = Field(
        ge=0, description="Total processing time in milliseconds"
    )

    @property
    def success_rate(self) -> float:
        """Calculate success rate of batch."""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if r.success)
        return successful / len(self.results)

    @property
    def by_symbol(self) -> dict[str, IndicatorResult]:
        """Get results indexed by symbol."""
        return {result.symbol: result for result in self.results}

    def get_errors(self) -> dict[str, list[str]]:
        """Get all errors by symbol."""
        return {
            result.symbol: result.errors for result in self.results if result.errors
        }


class ServiceHealth(BaseModel):
    """Service health status."""

    healthy: bool = Field(description="Overall health status")
    indicators_registered: int = Field(
        ge=0, description="Number of registered indicators"
    )
    cache_stats: CacheStats = Field(description="Cache statistics")
    active_symbols: int = Field(ge=0, description="Number of symbols with price data")
    last_calculation: datetime | None = Field(
        default=None, description="Timestamp of last calculation"
    )

    def to_summary(self) -> str:
        """Generate health summary."""
        status = "healthy" if self.healthy else "unhealthy"
        return (
            f"Service {status}: "
            f"{self.indicators_registered} indicators, "
            f"{self.active_symbols} active symbols, "
            f"{self.cache_stats.to_summary()}"
        )
