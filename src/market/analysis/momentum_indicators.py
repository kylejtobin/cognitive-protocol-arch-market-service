"""
Momentum indicators for market analysis.

This module provides momentum-based technical indicators that are composed
into Pydantic models for use in the analysis pipeline. Each indicator:
- Accepts validated input (prices, market data)
- Performs mathematical calculations
- Returns validated output with semantic interpretations
- Provides agent-friendly summaries
"""

from datetime import datetime
from decimal import Decimal
from typing import Literal

import pandas as pd
from pydantic import ConfigDict, Field, field_validator

from src.market.analysis.base import BaseIndicator
from src.market.analysis.contexts import (
    RSIAgentContext,
    RSIKeyLevels,
    RSISignals,
    SignalSuggestion,
)
from src.market.analysis.registry import register


@register("rsi")
class RSIAnalysis(BaseIndicator):
    """
    Relative Strength Index (RSI) analysis.

    RSI measures momentum by comparing the magnitude of recent gains
    to recent losses. Values range from 0-100, with:
    - Above 70: Overbought (potential reversal down)
    - Below 30: Oversold (potential reversal up)
    - Around 50: Neutral momentum
    """

    period: int = Field(default=14, ge=2, le=100)

    # RSI calculation results
    rsi_value: float = Field(ge=0, le=100)
    average_gain: float = Field(ge=0)
    average_loss: float = Field(ge=0)

    # Semantic interpretations
    momentum_state: Literal[
        "strongly_bullish", "bullish", "neutral", "bearish", "strongly_bearish"
    ]
    momentum_strength: Literal["extreme", "strong", "moderate", "weak"]
    is_overbought: bool = Field(default=False)
    is_oversold: bool = Field(default=False)

    # Divergence detection (optional)
    divergence_detected: bool = Field(default=False)
    divergence_type: Literal["bullish", "bearish", "none"] = Field(default="none")

    @field_validator("rsi_value")
    @classmethod
    def validate_rsi(cls, v: float) -> float:
        """Ensure RSI is within valid range."""
        if not 0 <= v <= 100:
            raise ValueError(f"RSI must be between 0 and 100, got {v}")
        return round(v, 2)

    @classmethod
    def from_price_series(
        cls,
        prices: pd.Series,
        symbol: str,
        timestamp: datetime,
        period: int = 14,
        check_divergence: bool = False,
    ) -> "RSIAnalysis":
        """
        Calculate RSI from a price series.

        Args:
            prices: Series of closing prices
            symbol: Trading symbol
            timestamp: Analysis timestamp
            period: RSI period (default: 14)
            check_divergence: Whether to check for divergence

        Returns:
            RSIAnalysis instance with calculations

        Raises:
            ValueError: If insufficient data for calculation

        """
        if len(prices) < period + 1:
            raise ValueError(
                f"Insufficient data: need at least {period + 1} prices, "
                f"got {len(prices)}"
            )

        # Calculate RSI using our own implementation
        rsi_values = cls._calculate_rsi(prices, period)
        current_rsi = float(rsi_values.iloc[-1])

        # Calculate average gains and losses for transparency
        price_changes = prices.diff()
        gains = price_changes.where(price_changes > 0, 0)
        losses = -price_changes.where(price_changes < 0, 0)

        # Use simple moving average for the most recent values
        avg_gain = float(gains.tail(period).mean())
        avg_loss = float(losses.tail(period).mean())

        # Determine momentum state
        momentum_state = cls._determine_momentum_state(current_rsi)
        momentum_strength = cls._determine_momentum_strength(current_rsi)

        # Check for divergence if requested
        divergence_detected = False
        divergence_type: Literal["bullish", "bearish", "none"] = "none"

        if check_divergence and len(prices) >= period * 2:
            divergence_detected, divergence_type = cls._check_divergence(
                prices, rsi_values, period
            )

        return cls(
            symbol=symbol,
            timestamp=timestamp,
            period=period,
            rsi_value=current_rsi,
            average_gain=avg_gain,
            average_loss=avg_loss,
            momentum_state=momentum_state,
            momentum_strength=momentum_strength,
            is_overbought=current_rsi >= 70,
            is_oversold=current_rsi <= 30,
            divergence_detected=divergence_detected,
            divergence_type=divergence_type,
        )

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate RSI values for a price series.

        Uses Wilder's smoothing method (exponential moving average).
        """
        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # First average: simple moving average
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()

        # Subsequent averages: Wilder's smoothing (EMA with alpha = 1/period)
        for i in range(period, len(prices)):
            if pd.notna(avg_gain.iloc[i - 1]):
                avg_gain.iloc[i] = (
                    avg_gain.iloc[i - 1] * (period - 1) + gains.iloc[i]
                ) / period
                avg_loss.iloc[i] = (
                    avg_loss.iloc[i - 1] * (period - 1) + losses.iloc[i]
                ) / period

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Handle division by zero (when avg_loss is 0)
        rsi = rsi.fillna(100)

        return rsi

    @staticmethod
    def _determine_momentum_state(
        rsi: float,
    ) -> Literal[
        "strongly_bullish", "bullish", "neutral", "bearish", "strongly_bearish"
    ]:
        """Determine momentum state from RSI value."""
        if rsi >= 80:
            return "strongly_bullish"
        elif rsi >= 60:
            return "bullish"
        elif rsi >= 40:
            return "neutral"
        elif rsi >= 20:
            return "bearish"
        else:
            return "strongly_bearish"

    @staticmethod
    def _determine_momentum_strength(
        rsi: float,
    ) -> Literal["extreme", "strong", "moderate", "weak"]:
        """Determine momentum strength from RSI value."""
        distance_from_center = abs(rsi - 50)
        if distance_from_center >= 40:  # RSI <= 10 or >= 90
            return "extreme"
        elif distance_from_center >= 30:  # RSI <= 20 or >= 80
            return "strong"
        elif distance_from_center >= 20:  # RSI <= 30 or >= 70
            return "moderate"
        else:
            return "weak"

    @staticmethod
    def _check_divergence(
        prices: pd.Series, rsi: pd.Series, period: int
    ) -> tuple[bool, Literal["bullish", "bearish", "none"]]:
        """
        Check for price/RSI divergence.

        Bullish divergence: Price makes lower low, RSI makes higher low
        Bearish divergence: Price makes higher high, RSI makes lower high
        """
        # Simple divergence check - can be enhanced
        recent_prices = prices.iloc[-period:]
        recent_rsi = rsi.iloc[-period:]

        # Find local extremes
        price_high_idx = recent_prices.idxmax()
        price_low_idx = recent_prices.idxmin()
        rsi_high_idx = recent_rsi.idxmax()
        rsi_low_idx = recent_rsi.idxmin()

        # Check for divergence patterns
        if price_high_idx != rsi_high_idx:
            # Potential bearish divergence
            if (
                recent_prices.iloc[-1] > recent_prices.iloc[0]
                and recent_rsi.iloc[-1] < recent_rsi.iloc[0]
            ):
                return True, "bearish"

        if price_low_idx != rsi_low_idx:
            # Potential bullish divergence
            if (
                recent_prices.iloc[-1] < recent_prices.iloc[0]
                and recent_rsi.iloc[-1] > recent_rsi.iloc[0]
            ):
                return True, "bullish"

        return False, "none"

    def semantic_summary(self) -> str:
        """Generate one-line summary for agent prompts."""
        state = self.momentum_state.replace("_", " ").title()

        if self.is_overbought:
            condition = "overbought"
        elif self.is_oversold:
            condition = "oversold"
        else:
            condition = f"{self.momentum_strength} momentum"

        summary = f"{state} momentum ({condition}) RSI:{self.rsi_value:.0f}"

        if self.divergence_detected:
            summary += f" with {self.divergence_type} divergence"

        return summary

    def to_agent_context(self) -> RSIAgentContext:
        """Format analysis for agent consumption."""
        interpretation = self._generate_interpretation()

        return RSIAgentContext(
            value=self.rsi_value,
            state=self.momentum_state,
            strength=self.momentum_strength,
            key_levels=RSIKeyLevels(
                current=self.rsi_value,
                overbought=70,
                oversold=30,
                neutral=50,
            ),
            signals=RSISignals(
                is_overbought=self.is_overbought,
                is_oversold=self.is_oversold,
                divergence=self.divergence_type if self.divergence_detected else None,
            ),
            interpretation=interpretation,
        )

    def _generate_interpretation(self) -> str:
        """Generate detailed interpretation for agents."""
        if self.is_overbought:
            return (
                f"RSI at {self.rsi_value:.1f} indicates overbought conditions. "
                "Market may be due for a pullback or consolidation. "
                "Consider taking profits or waiting for better entry."
            )
        elif self.is_oversold:
            return (
                f"RSI at {self.rsi_value:.1f} indicates oversold conditions. "
                "Potential bounce or reversal setup forming. "
                "Watch for bullish confirmation signals."
            )
        elif self.momentum_state == "bullish":
            return (
                f"RSI at {self.rsi_value:.1f} shows healthy bullish momentum. "
                "Trend favors upside continuation. "
                "Look for pullbacks to support for entry."
            )
        elif self.momentum_state == "bearish":
            return (
                f"RSI at {self.rsi_value:.1f} shows bearish momentum. "
                "Downtrend pressure persists. "
                "Rallies may offer shorting opportunities."
            )
        else:
            return (
                f"RSI at {self.rsi_value:.1f} indicates neutral momentum. "
                "Market lacks clear directional bias. "
                "Wait for momentum to develop before taking positions."
            )

    def _calculate_signal_strength(self) -> Literal["strong", "moderate", "weak"]:
        """Calculate trading signal strength based on RSI extremes."""
        if self.rsi_value >= 80 or self.rsi_value <= 20:
            return "strong"
        elif self.rsi_value >= 70 or self.rsi_value <= 30:
            return "moderate"
        else:
            return "weak"

    def suggest_signal(self) -> SignalSuggestion:
        """Suggest trading signal based on RSI analysis."""
        if self.is_oversold:
            return SignalSuggestion(
                bias="bullish",
                strength=self._calculate_signal_strength(),
                reason=f"RSI oversold at {self.rsi_value:.1f}",
                action="watch_for_reversal",
            )
        elif self.is_overbought:
            return SignalSuggestion(
                bias="bearish",
                strength=self._calculate_signal_strength(),
                reason=f"RSI overbought at {self.rsi_value:.1f}",
                action="consider_profit_taking",
            )
        elif self.divergence_detected:
            return SignalSuggestion(
                bias="bullish" if self.divergence_type == "bullish" else "bearish",
                strength="moderate",
                reason=f"{self.divergence_type.title()} divergence detected",
                action="await_confirmation",
            )
        else:
            return SignalSuggestion(
                bias="neutral",
                strength="weak",
                reason=f"No clear signal at RSI {self.rsi_value:.1f}",
                action="wait",
            )

    model_config = ConfigDict(
        json_encoders={
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat(),
        }
    )
