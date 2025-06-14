"""
MACD (Moving Average Convergence Divergence) indicator implementation.

This module provides a Pydantic-based MACD implementation that:
- Calculates MACD line, signal line, and histogram
- Provides semantic interpretations of trends
- Identifies crossovers and divergences
- Offers agent-friendly summaries with typed contexts
"""

from datetime import datetime
from decimal import Decimal
from typing import Literal

import pandas as pd
from pydantic import Field, ValidationInfo, computed_field, field_validator

from src.market.analysis.base import BaseIndicator
from src.market.analysis.contexts import (
    MACDAgentContext,
    MACDSignals,
    MACDValues,
    SignalSuggestion,
)
from src.market.analysis.registry import register
from src.market.domain.analysis_primitives import MACDValue


@register("macd")
class MACDAnalysis(BaseIndicator):
    """
    MACD (Moving Average Convergence Divergence) analysis.

    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of prices. It consists of:
    - MACD Line: 12-day EMA - 26-day EMA
    - Signal Line: 9-day EMA of MACD Line
    - Histogram: MACD Line - Signal Line
    """

    # MACD parameters
    fast_period: int = Field(default=12, ge=2, le=50)
    slow_period: int = Field(default=26, ge=5, le=200)
    signal_period: int = Field(default=9, ge=2, le=50)

    # Calculated values - store as Decimal for protocol compliance
    macd_value: Decimal = Field(description="MACD line value")
    signal_value: Decimal = Field(description="Signal line value")

    # Previous values for trend detection (optional)
    prev_macd: Decimal | None = Field(default=None)
    prev_signal: Decimal | None = Field(default=None)
    prev_histogram: Decimal | None = Field(default=None)

    # Semantic interpretations - computed from primitive
    @computed_field  # type: ignore[prop-decorator]
    @property
    def histogram_value(cls) -> Decimal:
        """Calculate histogram value."""
        return cls.macd_primitive.histogram

    @computed_field  # type: ignore[prop-decorator]
    @property
    def trend(cls) -> Literal["bullish", "bearish", "neutral"]:
        """Get trend from MACD primitive."""
        return cls.macd_primitive.trend

    @computed_field  # type: ignore[prop-decorator]
    @property
    def momentum(cls) -> Literal["increasing", "decreasing", "stable"]:
        """Determine momentum based on histogram changes."""
        if cls.prev_histogram is None:
            return "stable"

        current_hist = cls.histogram_value
        diff = abs(current_hist - cls.prev_histogram)
        if diff < Decimal("0.01"):
            return "stable"
        elif abs(current_hist) > abs(cls.prev_histogram):
            return "increasing"
        else:
            return "decreasing"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def signal_position(cls) -> Literal["above_signal", "below_signal", "at_signal"]:
        """Determine MACD position relative to signal line."""
        diff = abs(cls.macd_value - cls.signal_value)
        if diff < Decimal("0.01"):
            return "at_signal"
        elif cls.macd_value > cls.signal_value:
            return "above_signal"
        else:
            return "below_signal"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def histogram_trend(cls) -> Literal["expanding", "contracting", "neutral"]:
        """Determine if histogram is expanding or contracting."""
        if cls.prev_histogram is None:
            return "neutral"

        current = cls.histogram_value
        previous = cls.prev_histogram

        if abs(current) > abs(previous) * Decimal("1.1"):  # 10% threshold
            return "expanding"
        elif abs(current) < abs(previous) * Decimal("0.9"):
            return "contracting"
        else:
            return "neutral"

    # Crossover detection
    crossover_detected: bool = Field(default=False)
    crossover_type: Literal["bullish", "bearish", "none"] = Field(default="none")

    # Zero line analysis
    @computed_field  # type: ignore[prop-decorator]
    @property
    def above_zero(cls) -> bool:
        """Whether MACD is above zero line."""
        return cls.macd_primitive.is_positive

    zero_cross: Literal["bullish", "bearish", "none"] = Field(default="none")

    # Domain primitive as computed field
    @computed_field  # type: ignore[prop-decorator]
    @property
    def macd_primitive(cls) -> MACDValue:
        """Get MACD as domain primitive with rich behavior."""
        return MACDValue(macd_line=cls.macd_value, signal_line=cls.signal_value)

    @field_validator("slow_period")
    @classmethod
    def validate_slow_greater_than_fast(cls, v: int, info: ValidationInfo) -> int:
        """Ensure slow period is greater than fast period."""
        if "fast_period" in info.data and v <= info.data["fast_period"]:
            raise ValueError("Slow period must be greater than fast period")
        return v

    @classmethod
    def from_price_series(
        cls,
        prices: pd.Series,
        symbol: str,
        timestamp: datetime,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> "MACDAnalysis":
        """
        Calculate MACD from a price series.

        Args:
            prices: Series of closing prices
            symbol: Trading symbol
            timestamp: Analysis timestamp
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal EMA period (default: 9)

        Returns:
            MACDAnalysis instance with calculations

        Raises:
            ValueError: If insufficient data for calculation

        """
        min_required = slow_period + signal_period
        if len(prices) < min_required:
            raise ValueError(
                f"Insufficient data: need at least {min_required} prices, "
                f"got {len(prices)}"
            )

        # Calculate EMAs
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD line
        macd_line = ema_fast - ema_slow

        # Calculate signal line (EMA of MACD)
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Calculate histogram
        histogram = macd_line - signal_line

        # Get current and previous values
        current_macd = Decimal(str(macd_line.iloc[-1]))
        current_signal = Decimal(str(signal_line.iloc[-1]))

        # Previous values for trend detection
        prev_macd = Decimal(str(macd_line.iloc[-2])) if len(macd_line) > 1 else None
        prev_signal = (
            Decimal(str(signal_line.iloc[-2])) if len(signal_line) > 1 else None
        )
        prev_histogram = (
            Decimal(str(histogram.iloc[-2])) if len(histogram) > 1 else None
        )

        # Check for crossovers
        crossover_detected = False
        crossover_type: Literal["bullish", "bearish", "none"] = "none"
        if prev_macd is not None and prev_signal is not None:
            crossover_detected, crossover_type = cls._check_crossover(
                current_macd, current_signal, prev_macd, prev_signal
            )

        # Check zero line
        zero_cross = cls._check_zero_cross(current_macd, prev_macd)

        return cls(
            symbol=symbol,
            timestamp=timestamp,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            macd_value=current_macd,
            signal_value=current_signal,
            prev_macd=prev_macd,
            prev_signal=prev_signal,
            prev_histogram=prev_histogram,
            crossover_detected=crossover_detected,
            crossover_type=crossover_type,
            zero_cross=zero_cross,
        )

    @staticmethod
    def _check_crossover(
        current_macd: Decimal,
        current_signal: Decimal,
        prev_macd: Decimal,
        prev_signal: Decimal,
    ) -> tuple[bool, Literal["bullish", "bearish", "none"]]:
        """Check for MACD/Signal line crossover."""
        # Previous: MACD below signal, Current: MACD above signal = Bullish
        if prev_macd <= prev_signal and current_macd > current_signal:
            return True, "bullish"
        # Previous: MACD above signal, Current: MACD below signal = Bearish
        elif prev_macd >= prev_signal and current_macd < current_signal:
            return True, "bearish"
        else:
            return False, "none"

    @staticmethod
    def _check_zero_cross(
        current: Decimal, previous: Decimal | None
    ) -> Literal["bullish", "bearish", "none"]:
        """Check for zero line crossover."""
        if previous is None:
            return "none"
        # Crossed above zero
        if previous <= 0 and current > 0:
            return "bullish"
        # Crossed below zero
        elif previous >= 0 and current < 0:
            return "bearish"
        else:
            return "none"

    def semantic_summary(self) -> str:
        """Generate one-line summary for logging."""
        # Use primitive's format_display for consistent formatting
        primitive_display = self.macd_primitive.format_display()

        trend_str = self.trend.title()
        self.signal_position.replace("_", " ")

        summary = f"{trend_str} trend, {primitive_display}"

        if self.crossover_detected:
            summary += f" ({self.crossover_type} crossover)"
        elif self.histogram_trend == "expanding":
            summary += " (momentum expanding)"
        elif self.histogram_trend == "contracting":
            summary += " (momentum contracting)"

        return summary

    def to_agent_context(self) -> MACDAgentContext:
        """Format analysis for agent consumption."""
        interpretation = self._generate_interpretation()

        # Convert Decimal to float for the context
        macd_float = float(self.macd_value)
        signal_float = float(self.signal_value)
        histogram_float = float(self.histogram_value)

        return MACDAgentContext(
            values=MACDValues(
                macd=macd_float,
                signal=signal_float,
                histogram=histogram_float,
            ),
            trend=self.trend,
            momentum=self.momentum,
            signals=MACDSignals(
                histogram_trend=self.histogram_trend,
                signal_position=self.signal_position,
                zero_cross=self.zero_cross,
            ),
            interpretation=interpretation,
        )

    def _generate_interpretation(self) -> str:
        """Generate detailed interpretation for agents."""
        if self.crossover_detected:
            if self.crossover_type == "bullish":
                return (
                    "MACD bullish crossover detected. "
                    "Momentum shifting to the upside. "
                    "Consider long positions on confirmation."
                )
            else:
                return (
                    "MACD bearish crossover detected. "
                    "Momentum shifting to the downside. "
                    "Consider reducing long exposure."
                )
        elif self.trend == "bullish" and self.histogram_trend == "expanding":
            return (
                "Strong bullish momentum with expanding histogram. "
                "Trend acceleration in progress. "
                "Favorable for trend-following longs."
            )
        elif self.trend == "bearish" and self.histogram_trend == "expanding":
            return (
                "Strong bearish momentum with expanding histogram. "
                "Downtrend acceleration in progress. "
                "Avoid long positions."
            )
        elif self.histogram_trend == "contracting":
            return (
                f"Momentum contracting in {self.trend} trend. "
                "Potential trend exhaustion or reversal ahead. "
                "Monitor for directional change."
            )
        else:
            return (
                f"MACD shows {self.trend} bias with {self.momentum} momentum. "
                f"Currently {self.signal_position.replace('_', ' ')}. "
                "No strong directional signal."
            )

    def suggest_signal(self) -> SignalSuggestion:
        """Suggest trading signal based on MACD analysis."""
        if self.crossover_detected:
            return SignalSuggestion(
                bias="bullish" if self.crossover_type == "bullish" else "bearish",
                strength="strong",
                reason=f"MACD {self.crossover_type} crossover",
                action="enter_on_confirmation"
                if self.crossover_type == "bullish"
                else "exit_longs",
            )
        elif self.zero_cross != "none":
            return SignalSuggestion(
                bias="bullish" if self.zero_cross == "bullish" else "bearish",
                strength="moderate",
                reason=f"MACD crossed {'above' if self.zero_cross == 'bullish' else 'below'} zero",  # noqa: E501
                action="prepare_entry"
                if self.zero_cross == "bullish"
                else "reduce_risk",
            )
        elif self.trend == "bullish" and self.histogram_trend == "expanding":
            return SignalSuggestion(
                bias="bullish",
                strength="moderate",
                reason="Bullish momentum expanding",
                action="hold_or_add",
            )
        elif self.trend == "bearish" and self.histogram_trend == "expanding":
            return SignalSuggestion(
                bias="bearish",
                strength="moderate",
                reason="Bearish momentum expanding",
                action="stay_out_or_short",
            )
        else:
            return SignalSuggestion(
                bias="neutral",
                strength="weak",
                reason=f"No clear MACD signal, {self.momentum} momentum",
                action="wait_for_clarity",
            )
