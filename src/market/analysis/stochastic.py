"""
Stochastic oscillator indicator.

The Stochastic oscillator is a momentum indicator that shows the location
of the close relative to the high-low range over a set number of periods.
"""

from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import Field

from src.market.analysis.base import BaseIndicator
from src.market.analysis.contexts import (
    SignalSuggestion,
    StochasticAgentContext,
    StochasticValues,
    StochasticZones,
)
from src.market.analysis.registry import register


@register("stochastic")
class StochasticAnalysis(BaseIndicator):
    """
    Stochastic oscillator analysis.

    The indicator consists of two lines:
    - %K: The main line showing current close relative to high-low range
    - %D: A moving average of %K (signal line)

    Values range from 0-100:
    - Above 80: Overbought
    - Below 20: Oversold
    """

    # Configuration
    k_period: int = Field(default=14, ge=3, le=50, description="Lookback period for %K")
    k_smooth: int = Field(default=3, ge=1, le=10, description="Smoothing period for %K")
    d_smooth: int = Field(default=3, ge=1, le=10, description="Smoothing period for %D")

    # Stochastic values
    k_value: float = Field(ge=0, le=100, description="%K value")
    d_value: float = Field(ge=0, le=100, description="%D signal line value")

    # Semantic interpretations
    momentum_zone: Literal["overbought", "oversold", "neutral"]
    crossover_detected: bool = Field(default=False)
    crossover_type: Literal["bullish", "bearish", "none"] = Field(default="none")
    divergence_detected: bool = Field(default=False)
    divergence_type: Literal["bullish", "bearish", "none"] = Field(default="none")

    @classmethod
    def from_price_series(
        cls,
        prices: pd.Series,
        symbol: str,
        timestamp: datetime,
        k_period: int = 14,
        k_smooth: int = 3,
        d_smooth: int = 3,
        high_prices: pd.Series | None = None,
        low_prices: pd.Series | None = None,
        check_divergence: bool = False,
    ) -> "StochasticAnalysis":
        """
        Calculate Stochastic oscillator from price series.

        Args:
            prices: Series of closing prices
            symbol: Trading symbol
            timestamp: Analysis timestamp
            k_period: Lookback period for %K (default: 14)
            k_smooth: Smoothing period for %K (default: 3)
            d_smooth: Smoothing period for %D (default: 3)
            high_prices: Series of high prices (optional, uses close if not provided)
            low_prices: Series of low prices (optional, uses close if not provided)
            check_divergence: Whether to check for divergence

        Returns:
            StochasticAnalysis instance

        Raises:
            ValueError: If insufficient data

        """
        min_required = k_period + k_smooth + d_smooth
        if len(prices) < min_required:
            raise ValueError(
                f"Insufficient data: need at least {min_required} prices, "
                f"got {len(prices)}"
            )

        # Use close prices for high/low if not provided
        if high_prices is None:
            high_prices = prices
        if low_prices is None:
            low_prices = prices

        # Ensure all series have same length
        min_len = min(len(prices), len(high_prices), len(low_prices))
        prices = prices.iloc[-min_len:]
        high_prices = high_prices.iloc[-min_len:]
        low_prices = low_prices.iloc[-min_len:]

        # Calculate raw %K
        lowest_low = low_prices.rolling(window=k_period).min()
        highest_high = high_prices.rolling(window=k_period).max()

        # Avoid division by zero
        range_diff = highest_high - lowest_low
        range_diff = range_diff.replace(
            0, 1
        )  # Replace 0 with 1 to avoid division by zero

        raw_k = ((prices - lowest_low) / range_diff) * 100

        # Smooth %K
        k = raw_k.rolling(window=k_smooth).mean()

        # Calculate %D (signal line)
        d = k.rolling(window=d_smooth).mean()

        # Get current values
        current_k = float(k.iloc[-1])
        current_d = float(d.iloc[-1])

        # Check for crossover
        crossover_detected, crossover_type = cls._check_crossover(k.tail(3), d.tail(3))

        # Determine momentum zone
        momentum_zone = cls._determine_momentum_zone(current_k)

        # Check for divergence if requested
        divergence_detected = False
        divergence_type: Literal["bullish", "bearish", "none"] = "none"

        if check_divergence and len(prices) >= k_period * 2:
            divergence_detected, divergence_type = cls._check_divergence(
                prices, k, k_period
            )

        return cls(
            symbol=symbol,
            timestamp=timestamp,
            k_period=k_period,
            k_smooth=k_smooth,
            d_smooth=d_smooth,
            k_value=current_k,
            d_value=current_d,
            momentum_zone=momentum_zone,
            crossover_detected=crossover_detected,
            crossover_type=crossover_type,
            divergence_detected=divergence_detected,
            divergence_type=divergence_type,
        )

    @staticmethod
    def _check_crossover(
        k_series: pd.Series, d_series: pd.Series
    ) -> tuple[bool, Literal["bullish", "bearish", "none"]]:
        """Check for %K/%D crossover."""
        if len(k_series) < 2 or len(d_series) < 2:
            return False, "none"

        # Previous and current positions
        prev_diff = float(k_series.iloc[-2] - d_series.iloc[-2])
        curr_diff = float(k_series.iloc[-1] - d_series.iloc[-1])

        # Check for crossover
        if prev_diff <= 0 < curr_diff:
            return True, "bullish"
        elif prev_diff >= 0 > curr_diff:
            return True, "bearish"
        else:
            return False, "none"

    @staticmethod
    def _determine_momentum_zone(
        k_value: float,
    ) -> Literal["overbought", "oversold", "neutral"]:
        """Determine momentum zone from %K value."""
        if k_value >= 80:
            return "overbought"
        elif k_value <= 20:
            return "oversold"
        else:
            return "neutral"

    @staticmethod
    def _check_divergence(
        prices: pd.Series, k_series: pd.Series, period: int
    ) -> tuple[bool, Literal["bullish", "bearish", "none"]]:
        """
        Check for price/stochastic divergence.

        Bullish divergence: Price makes lower low, Stochastic makes higher low
        Bearish divergence: Price makes higher high, Stochastic makes lower high
        """
        recent_prices = prices.tail(period)
        recent_k = k_series.tail(period)

        # Find peaks and troughs
        price_peaks = []
        k_peaks = []
        price_troughs = []
        k_troughs = []

        for i in range(1, len(recent_prices) - 1):
            # Peaks
            if (
                recent_prices.iloc[i] > recent_prices.iloc[i - 1]
                and recent_prices.iloc[i] > recent_prices.iloc[i + 1]
            ):
                price_peaks.append(i)

            if (
                recent_k.iloc[i] > recent_k.iloc[i - 1]
                and recent_k.iloc[i] > recent_k.iloc[i + 1]
            ):
                k_peaks.append(i)

            # Troughs
            if (
                recent_prices.iloc[i] < recent_prices.iloc[i - 1]
                and recent_prices.iloc[i] < recent_prices.iloc[i + 1]
            ):
                price_troughs.append(i)

            if (
                recent_k.iloc[i] < recent_k.iloc[i - 1]
                and recent_k.iloc[i] < recent_k.iloc[i + 1]
            ):
                k_troughs.append(i)

        # Check for bearish divergence
        if len(price_peaks) >= 2 and len(k_peaks) >= 2:
            if (
                recent_prices.iloc[price_peaks[-1]]
                > recent_prices.iloc[price_peaks[-2]]
                and recent_k.iloc[k_peaks[-1]] < recent_k.iloc[k_peaks[-2]]
            ):
                return True, "bearish"

        # Check for bullish divergence
        if len(price_troughs) >= 2 and len(k_troughs) >= 2:
            if (
                recent_prices.iloc[price_troughs[-1]]
                < recent_prices.iloc[price_troughs[-2]]
                and recent_k.iloc[k_troughs[-1]] > recent_k.iloc[k_troughs[-2]]
            ):
                return True, "bullish"

        return False, "none"

    def semantic_summary(self) -> str:
        """Generate one-line summary for agent prompts."""
        zone_desc = self.momentum_zone.replace("_", " ")

        summary = f"Stochastic {zone_desc} (K:{self.k_value:.1f}, D:{self.d_value:.1f})"

        if self.crossover_detected:
            summary += f" - {self.crossover_type} crossover"

        if self.divergence_detected:
            summary += f" with {self.divergence_type} divergence"

        return summary

    def to_agent_context(self) -> StochasticAgentContext:
        """Format analysis for agent consumption."""
        # Determine momentum based on crossover and zone
        momentum: Literal["bullish", "bearish", "neutral"]
        if self.crossover_type == "bullish" or self.momentum_zone == "oversold":
            momentum = "bullish"
        elif self.crossover_type == "bearish" or self.momentum_zone == "overbought":
            momentum = "bearish"
        else:
            momentum = "neutral"

        return StochasticAgentContext(
            values=StochasticValues(
                k_percent=self.k_value,
                d_percent=self.d_value,
            ),
            zones=StochasticZones(
                current_zone=self.momentum_zone,
                k_zone=self.momentum_zone,
                d_zone=self._determine_momentum_zone(self.d_value),
            ),
            momentum=momentum,
            crossover=self.crossover_type,
            interpretation=self._generate_interpretation(),
        )

    def _generate_interpretation(self) -> str:
        """Generate detailed interpretation for agents."""
        if self.crossover_detected:
            if self.crossover_type == "bullish" and self.momentum_zone == "oversold":
                return (
                    "Bullish stochastic crossover in oversold territory. "
                    "Strong buy signal - momentum turning positive from extreme lows. "
                    "Consider long entry with tight stop below recent low."
                )
            elif (
                self.crossover_type == "bearish" and self.momentum_zone == "overbought"
            ):
                return (
                    "Bearish stochastic crossover in overbought territory. "
                    "Strong sell signal - momentum turning negative from extreme highs."
                    "Consider taking profits or initiating short positions."
                )
            elif self.crossover_type == "bullish":
                return (
                    f"Bullish crossover at {self.k_value:.1f}. "
                    "Momentum turning positive, potential entry opportunity. "
                    "Confirm with price action and volume."
                )
            else:
                return (
                    f"Bearish crossover at {self.k_value:.1f}. "
                    "Momentum turning negative, exercise caution. "
                    "Watch for support levels."
                )
        elif self.divergence_detected:
            if self.divergence_type == "bullish":
                return (
                    "Bullish divergence detected - price making lower lows but stochastic higher lows. "  # noqa: E501
                    "Potential reversal setup forming. "
                    "Watch for bullish confirmation and crossover signal."
                )
            else:
                return (
                    "Bearish divergence detected - price making higher highs but stochastic lower highs. "  # noqa: E501
                    "Momentum not confirming price advance. "
                    "Potential top forming, consider reducing exposure."
                )
        elif self.momentum_zone == "overbought":
            return (
                f"Stochastic at {self.k_value:.1f} in overbought zone. "
                "Market may be overextended to upside. "
                "Wait for bearish crossover or pullback for better entry."
            )
        elif self.momentum_zone == "oversold":
            return (
                f"Stochastic at {self.k_value:.1f} in oversold zone. "
                "Market may be overextended to downside. "
                "Watch for bullish crossover or bounce opportunity."
            )
        else:
            return (
                f"Stochastic at {self.k_value:.1f} in neutral zone. "
                "No extreme conditions present. "
                "Wait for clearer signals or zone entry."
            )

    def suggest_signal(self) -> SignalSuggestion:
        """Suggest trading signal based on stochastic analysis."""
        if self.crossover_detected:
            if self.crossover_type == "bullish" and self.momentum_zone == "oversold":
                return SignalSuggestion(
                    bias="bullish",
                    strength="strong",
                    reason="Bullish crossover in oversold zone",
                    action="enter_long",
                )
            elif (
                self.crossover_type == "bearish" and self.momentum_zone == "overbought"
            ):
                return SignalSuggestion(
                    bias="bearish",
                    strength="strong",
                    reason="Bearish crossover in overbought zone",
                    action="enter_short",
                )
            else:
                bias: Literal["bullish", "bearish", "neutral"] = (
                    "bullish" if self.crossover_type == "bullish" else "bearish"
                )
                return SignalSuggestion(
                    bias=bias,
                    strength="moderate",
                    reason=f"{self.crossover_type.title()} crossover at {self.k_value:.1f}",  # noqa: E501
                    action="await_confirmation",
                )
        elif self.divergence_detected:
            div_bias: Literal["bullish", "bearish", "neutral"] = (
                "bullish" if self.divergence_type == "bullish" else "bearish"
            )
            return SignalSuggestion(
                bias=div_bias,
                strength="moderate",
                reason=f"{self.divergence_type.title()} divergence detected",
                action="prepare_reversal",
            )
        elif self.momentum_zone == "oversold":
            return SignalSuggestion(
                bias="bullish",
                strength="weak",
                reason=f"Oversold at {self.k_value:.1f}",
                action="watch_for_bounce",
            )
        elif self.momentum_zone == "overbought":
            return SignalSuggestion(
                bias="bearish",
                strength="weak",
                reason=f"Overbought at {self.k_value:.1f}",
                action="watch_for_pullback",
            )
        else:
            return SignalSuggestion(
                bias="neutral",
                strength="weak",
                reason="No clear stochastic signal",
                action="wait",
            )
