"""
MACD (Moving Average Convergence Divergence) indicator.

This module provides the MACD momentum indicator for trend analysis.
MACD shows the relationship between two moving averages of prices.
"""

from datetime import datetime
from typing import Any, Literal

import pandas as pd
from pydantic import Field, ValidationInfo, field_validator

from src.market.analysis.base import BaseIndicator
from src.market.analysis.registry import register


@register("macd")
class MACDAnalysis(BaseIndicator):
    """
    MACD (Moving Average Convergence Divergence) analysis.

    MACD is calculated by subtracting the 26-period EMA from the 12-period EMA.
    A 9-period EMA of the MACD (signal line) is then plotted on top.

    Signals:
    - MACD crosses above signal: Bullish signal
    - MACD crosses below signal: Bearish signal
    - Divergence from price: Potential reversal
    """

    # Configuration
    fast_period: int = Field(default=12, ge=2, le=50)
    slow_period: int = Field(default=26, ge=10, le=100)
    signal_period: int = Field(default=9, ge=2, le=50)

    # MACD values
    macd_line: float = Field(description="MACD line value (fast EMA - slow EMA)")
    signal_line: float = Field(description="Signal line value (EMA of MACD)")
    histogram: float = Field(description="MACD histogram (MACD - Signal)")

    # Semantic interpretations
    trend_state: Literal["bullish", "bearish", "neutral"]
    trend_strength: Literal["strong", "moderate", "weak"]
    crossover_detected: bool = Field(default=False)
    crossover_type: Literal["bullish", "bearish", "none"] = Field(default="none")

    # Divergence detection
    divergence_detected: bool = Field(default=False)
    divergence_type: Literal["bullish", "bearish", "none"] = Field(default="none")

    @field_validator("slow_period")
    @classmethod
    def validate_slow_period(cls, v: int, info: ValidationInfo) -> int:
        """Ensure slow period is greater than fast period."""
        fast = info.data.get("fast_period", 12)
        if v <= fast:
            raise ValueError(
                f"Slow period ({v}) must be greater than fast period ({fast})"
            )
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
        check_divergence: bool = False,
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
            check_divergence: Whether to check for divergence

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
        macd = ema_fast - ema_slow

        # Calculate signal line (EMA of MACD)
        signal = macd.ewm(span=signal_period, adjust=False).mean()

        # Calculate histogram
        histogram = macd - signal

        # Get current values
        current_macd = float(macd.iloc[-1])
        current_signal = float(signal.iloc[-1])
        current_histogram = float(histogram.iloc[-1])

        # Check for crossover in recent data
        crossover_detected, crossover_type = cls._check_crossover(
            macd.tail(3), signal.tail(3)
        )

        # Determine trend state and strength
        trend_state = cls._determine_trend_state(current_macd, current_histogram)
        trend_strength = cls._determine_trend_strength(current_histogram, histogram)

        # Check for divergence if requested
        divergence_detected = False
        divergence_type: Literal["bullish", "bearish", "none"] = "none"

        if check_divergence and len(prices) >= slow_period * 2:
            divergence_detected, divergence_type = cls._check_divergence(
                prices, macd, slow_period
            )

        return cls(
            symbol=symbol,
            timestamp=timestamp,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            macd_line=current_macd,
            signal_line=current_signal,
            histogram=current_histogram,
            trend_state=trend_state,
            trend_strength=trend_strength,
            crossover_detected=crossover_detected,
            crossover_type=crossover_type,
            divergence_detected=divergence_detected,
            divergence_type=divergence_type,
        )

    @staticmethod
    def _check_crossover(
        macd: pd.Series, signal: pd.Series
    ) -> tuple[bool, Literal["bullish", "bearish", "none"]]:
        """Check for MACD/Signal line crossover."""
        if len(macd) < 2 or len(signal) < 2:
            return False, "none"

        # Previous and current positions
        prev_diff = float(macd.iloc[-2] - signal.iloc[-2])
        curr_diff = float(macd.iloc[-1] - signal.iloc[-1])

        # Check for crossover
        if prev_diff <= 0 < curr_diff:
            return True, "bullish"
        elif prev_diff >= 0 > curr_diff:
            return True, "bearish"
        else:
            return False, "none"

    @staticmethod
    def _determine_trend_state(
        macd: float, histogram: float
    ) -> Literal["bullish", "bearish", "neutral"]:
        """Determine trend state from MACD values."""
        if macd > 0 and histogram > 0:
            return "bullish"
        elif macd < 0 and histogram < 0:
            return "bearish"
        else:
            return "neutral"

    @staticmethod
    def _determine_trend_strength(
        current_histogram: float, histogram_series: pd.Series
    ) -> Literal["strong", "moderate", "weak"]:
        """Determine trend strength from histogram."""
        # Use recent histogram values for strength assessment
        recent_histogram = histogram_series.tail(10)
        avg_histogram = float(recent_histogram.abs().mean())

        if abs(current_histogram) > avg_histogram * 1.5:
            return "strong"
        elif abs(current_histogram) > avg_histogram * 0.5:
            return "moderate"
        else:
            return "weak"

    @staticmethod
    def _check_divergence(
        prices: pd.Series, macd: pd.Series, period: int
    ) -> tuple[bool, Literal["bullish", "bearish", "none"]]:
        """
        Check for price/MACD divergence.

        Bullish divergence: Price makes lower low, MACD makes higher low
        Bearish divergence: Price makes higher high, MACD makes lower high
        """
        recent_prices = prices.tail(period)
        recent_macd = macd.tail(period)

        # Find peaks and troughs
        price_peaks = []
        macd_peaks = []

        for i in range(1, len(recent_prices) - 1):
            # Price peaks
            if (
                recent_prices.iloc[i] > recent_prices.iloc[i - 1]
                and recent_prices.iloc[i] > recent_prices.iloc[i + 1]
            ):
                price_peaks.append(i)

            # MACD peaks
            if (
                recent_macd.iloc[i] > recent_macd.iloc[i - 1]
                and recent_macd.iloc[i] > recent_macd.iloc[i + 1]
            ):
                macd_peaks.append(i)

        # Check for divergence patterns
        if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
            # Bearish divergence: higher price high, lower MACD high
            if (
                recent_prices.iloc[price_peaks[-1]]
                > recent_prices.iloc[price_peaks[-2]]
                and recent_macd.iloc[macd_peaks[-1]] < recent_macd.iloc[macd_peaks[-2]]
            ):
                return True, "bearish"

        # Similar check for bullish divergence (lower lows)
        price_troughs = []
        macd_troughs = []

        for i in range(1, len(recent_prices) - 1):
            if (
                recent_prices.iloc[i] < recent_prices.iloc[i - 1]
                and recent_prices.iloc[i] < recent_prices.iloc[i + 1]
            ):
                price_troughs.append(i)

            if (
                recent_macd.iloc[i] < recent_macd.iloc[i - 1]
                and recent_macd.iloc[i] < recent_macd.iloc[i + 1]
            ):
                macd_troughs.append(i)

        if len(price_troughs) >= 2 and len(macd_troughs) >= 2:
            # Bullish divergence: lower price low, higher MACD low
            if (
                recent_prices.iloc[price_troughs[-1]]
                < recent_prices.iloc[price_troughs[-2]]
                and recent_macd.iloc[macd_troughs[-1]]
                > recent_macd.iloc[macd_troughs[-2]]
            ):
                return True, "bullish"

        return False, "none"

    def semantic_summary(self) -> str:
        """Generate one-line summary for agent prompts."""
        state = self.trend_state.title()

        if self.crossover_detected:
            summary = f"MACD {self.crossover_type} crossover - {state} trend"
        else:
            summary = f"MACD {state} ({self.trend_strength})"

        if self.histogram > 0:
            summary += f" momentum increasing ({self.histogram:.3f})"
        else:
            summary += f" momentum decreasing ({self.histogram:.3f})"

        if self.divergence_detected:
            summary += f" with {self.divergence_type} divergence"

        return summary

    def to_agent_context(self) -> dict[str, Any]:
        """Format analysis for agent consumption."""
        return {
            "indicator": "macd",
            "values": {
                "macd": round(self.macd_line, 4),
                "signal": round(self.signal_line, 4),
                "histogram": round(self.histogram, 4),
            },
            "state": self.trend_state,
            "strength": self.trend_strength,
            "signals": {
                "crossover": self.crossover_type if self.crossover_detected else None,
                "divergence": self.divergence_type
                if self.divergence_detected
                else None,
                "histogram_direction": "positive" if self.histogram > 0 else "negative",
            },
            "interpretation": self._generate_interpretation(),
        }

    def _generate_interpretation(self) -> str:
        """Generate detailed interpretation for agents."""
        if self.crossover_detected:
            if self.crossover_type == "bullish":
                return (
                    "MACD bullish crossover detected. Momentum shifting to upside. "
                    "Consider long positions or adding to existing longs. "
                    "Confirm with price action and volume."
                )
            else:
                return (
                    "MACD bearish crossover detected. Momentum shifting to downside. "
                    "Consider reducing long exposure or initiating shorts. "
                    "Watch for support levels."
                )
        elif self.divergence_detected:
            if self.divergence_type == "bullish":
                return (
                    "Bullish divergence detected between price and MACD. "
                    "Potential bottom forming despite lower prices. "
                    "Watch for reversal confirmation."
                )
            else:
                return (
                    "Bearish divergence detected between price and MACD. "
                    "Momentum weakening despite higher prices. "
                    "Potential top forming, consider taking profits."
                )
        elif self.trend_state == "bullish" and self.trend_strength == "strong":
            return (
                "Strong bullish momentum confirmed by MACD. "
                "Trend favors continuation higher. "
                "Use pullbacks to MACD line as entry opportunities."
            )
        elif self.trend_state == "bearish" and self.trend_strength == "strong":
            return (
                "Strong bearish momentum confirmed by MACD. "
                "Downtrend likely to continue. "
                "Rallies to MACD line offer shorting opportunities."
            )
        else:
            return (
                f"MACD showing {self.trend_strength} {self.trend_state} momentum. "
                "No clear directional signal. "
                "Wait for stronger momentum or crossover signal."
            )

    def suggest_signal(self) -> dict[str, str]:
        """Suggest trading signal based on MACD analysis."""
        if self.crossover_detected:
            return {
                "bias": self.crossover_type,
                "strength": "strong" if self.trend_strength == "strong" else "moderate",
                "reason": f"MACD {self.crossover_type} crossover",
                "action": "enter_position"
                if self.trend_strength == "strong"
                else "await_confirmation",
            }
        elif self.divergence_detected:
            return {
                "bias": self.divergence_type,
                "strength": "moderate",
                "reason": f"{self.divergence_type.title()} divergence detected",
                "action": "prepare_reversal_trade",
            }
        elif self.trend_state != "neutral" and self.trend_strength == "strong":
            return {
                "bias": self.trend_state,
                "strength": "moderate",
                "reason": f"Strong {self.trend_state} momentum",
                "action": "follow_trend",
            }
        else:
            return {
                "bias": "neutral",
                "strength": "weak",
                "reason": "No clear MACD signal",
                "action": "wait",
            }
