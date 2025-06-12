"""
Volume Analysis indicator.

This module provides volume-based market analysis including VWAP,
volume trends, buy/sell pressure estimation, and volume profile analysis.
"""

from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from src.market.analysis.base import BaseIndicator
from src.market.analysis.registry import register


@register("volume")
class VolumeAnalysis(BaseIndicator):
    """
    Volume analysis for market activity and liquidity assessment.

    Provides insights into:
    - Volume Weighted Average Price (VWAP)
    - Volume trends and momentum
    - Buy/sell pressure estimation
    - Price-volume correlations
    - Volume profile for support/resistance
    """

    # Configuration
    period: int = Field(default=14, ge=2, le=100)

    # Volume metrics
    current_volume: float = Field(description="Current period volume")
    average_volume: float = Field(description="Average volume over period")
    vwap: float = Field(description="Volume Weighted Average Price")
    volume_ratio: float = Field(ge=0, description="Current/Average volume ratio")

    # Buy/Sell pressure
    buy_volume_pct: float = Field(ge=0, le=1, description="Percentage of buy volume")
    sell_volume_pct: float = Field(ge=0, le=1, description="Percentage of sell volume")

    # Volume characteristics
    volume_trend: Literal["increasing", "decreasing", "stable"]
    volume_strength: Literal["high", "moderate", "low"]
    price_volume_correlation: float = Field(ge=-1, le=1)

    # Optional volume profile data
    volume_nodes: list[dict[str, float]] | None = Field(
        default=None, description="High volume price levels for support/resistance"
    )

    @model_validator(mode="after")
    def validate_buy_sell_sum(self) -> "VolumeAnalysis":
        """Ensure buy and sell percentages sum to 1."""
        if abs((self.buy_volume_pct + self.sell_volume_pct) - 1.0) > 0.01:
            raise ValueError(
                f"Buy ({self.buy_volume_pct}) and sell ({self.sell_volume_pct}) "
                f"percentages must sum to 1.0"
            )
        return self

    @classmethod
    def from_price_volume_series(
        cls,
        prices: pd.Series,
        volumes: pd.Series,
        symbol: str,
        timestamp: datetime,
        period: int = 14,
        calculate_profile: bool = False,
    ) -> "VolumeAnalysis":
        """
        Calculate volume analysis from price and volume series.

        Args:
            prices: Series of prices
            volumes: Series of volumes
            symbol: Trading symbol
            timestamp: Analysis timestamp
            period: Period for calculations
            calculate_profile: Whether to calculate volume profile

        Returns:
            VolumeAnalysis instance

        """
        if len(prices) < period or len(volumes) < period:
            raise ValueError(
                f"Insufficient data: need at least {period} data points, "
                f"got {min(len(prices), len(volumes))}"
            )

        # Ensure same length
        min_len = min(len(prices), len(volumes))
        prices = prices.iloc[-min_len:]
        volumes = volumes.iloc[-min_len:]

        # Calculate VWAP
        vwap = float((prices * volumes).sum() / volumes.sum())

        # Current and average volume
        current_volume = float(volumes.iloc[-1])
        average_volume = float(volumes.mean())
        volume_ratio = current_volume / average_volume if average_volume > 0 else 1.0

        # Volume trend
        recent_volumes = volumes.tail(period)
        volume_trend = cls._determine_volume_trend(recent_volumes)

        # Volume strength
        volume_strength = cls._determine_volume_strength(volume_ratio)

        # Price-volume correlation
        price_changes = prices.pct_change().dropna()
        volume_changes = volumes.pct_change().dropna()
        min_corr_len = min(len(price_changes), len(volume_changes))

        if min_corr_len >= 3:
            correlation = float(
                price_changes.tail(min_corr_len).corr(volume_changes.tail(min_corr_len))
            )
            # Handle NaN correlation (e.g., when all values are the same)
            if pd.isna(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        # Estimate buy/sell pressure (simplified without trade data)
        buy_pct, sell_pct = cls._estimate_buy_sell_pressure(prices, volumes)

        # Volume profile if requested
        volume_nodes = None
        if calculate_profile:
            volume_nodes = cls._calculate_volume_profile(prices, volumes)

        return cls(
            symbol=symbol,
            timestamp=timestamp,
            period=period,
            current_volume=current_volume,
            average_volume=average_volume,
            vwap=vwap,
            volume_ratio=volume_ratio,
            buy_volume_pct=buy_pct,
            sell_volume_pct=sell_pct,
            volume_trend=volume_trend,
            volume_strength=volume_strength,
            price_volume_correlation=correlation,
            volume_nodes=volume_nodes,
        )

    @classmethod
    def from_trades(
        cls,
        trades: pd.DataFrame,
        symbol: str,
        timestamp: datetime,
        period: int = 14,
    ) -> "VolumeAnalysis":
        """
        Calculate volume analysis from trade data.

        Args:
            trades: DataFrame with columns: price, size, side
            symbol: Trading symbol
            timestamp: Analysis timestamp
            period: Period for calculations

        Returns:
            VolumeAnalysis instance

        """
        if len(trades) < period:
            raise ValueError(
                f"Insufficient data: need at least {period} trades, got {len(trades)}"
            )

        # Calculate aggregated metrics
        prices = trades["price"]
        volumes = trades["size"]

        # VWAP
        vwap = float((prices * volumes).sum() / volumes.sum())

        # Current and average volume
        current_volume = float(volumes.iloc[-1])
        average_volume = float(volumes.mean())
        volume_ratio = current_volume / average_volume

        # Buy/sell pressure from trade sides
        buy_volume = trades[trades["side"] == "buy"]["size"].sum()
        sell_volume = trades[trades["side"] == "sell"]["size"].sum()
        total_volume = buy_volume + sell_volume

        buy_pct = float(buy_volume / total_volume) if total_volume > 0 else 0.5
        sell_pct = float(sell_volume / total_volume) if total_volume > 0 else 0.5

        # Volume trend
        volume_by_period = volumes.rolling(max(1, len(volumes) // period)).sum()
        volume_trend = cls._determine_volume_trend(volume_by_period.dropna())

        # Volume strength
        volume_strength = cls._determine_volume_strength(volume_ratio)

        # Price-volume correlation
        price_series = pd.Series(prices.values)
        volume_series = pd.Series(volumes.values)
        correlation = float(price_series.corr(volume_series))

        return cls(
            symbol=symbol,
            timestamp=timestamp,
            period=period,
            current_volume=current_volume,
            average_volume=average_volume,
            vwap=vwap,
            volume_ratio=volume_ratio,
            buy_volume_pct=buy_pct,
            sell_volume_pct=sell_pct,
            volume_trend=volume_trend,
            volume_strength=volume_strength,
            price_volume_correlation=correlation,
        )

    @staticmethod
    def _determine_volume_trend(
        volumes: pd.Series,
    ) -> Literal["increasing", "decreasing", "stable"]:
        """Determine volume trend from recent volumes."""
        if len(volumes) < 3:
            return "stable"

        # Linear regression on volume
        x = np.arange(len(volumes))
        slope = np.polyfit(x, volumes.values, 1)[0]

        # Normalize slope by average volume
        avg_volume = volumes.mean()
        normalized_slope = slope / avg_volume if avg_volume > 0 else 0

        if normalized_slope > 0.05:
            return "increasing"
        elif normalized_slope < -0.05:
            return "decreasing"
        else:
            return "stable"

    @staticmethod
    def _determine_volume_strength(ratio: float) -> Literal["high", "moderate", "low"]:
        """Classify volume strength based on ratio to average."""
        if ratio >= 1.5:
            return "high"
        elif ratio >= 0.7:
            return "moderate"
        else:
            return "low"

    @staticmethod
    def _estimate_buy_sell_pressure(
        prices: pd.Series, volumes: pd.Series
    ) -> tuple[float, float]:
        """
        Estimate buy/sell pressure from price and volume movements.

        This is a simplified estimation when trade-level data isn't available.
        """
        # Use price direction to estimate buy/sell volume
        price_changes = prices.diff()

        buy_volume = volumes[price_changes > 0].sum()
        sell_volume = volumes[price_changes <= 0].sum()
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            return 0.5, 0.5

        buy_pct = float(buy_volume / total_volume)
        sell_pct = float(sell_volume / total_volume)

        # Ensure they sum to 1
        total = buy_pct + sell_pct
        if total > 0:
            buy_pct /= total
            sell_pct /= total

        return buy_pct, sell_pct

    @staticmethod
    def _calculate_volume_profile(
        prices: pd.Series, volumes: pd.Series, n_bins: int = 10
    ) -> list[dict[str, float]]:
        """Calculate volume profile to identify high volume price levels."""
        # Create price bins
        price_min, price_max = prices.min(), prices.max()
        bins = np.linspace(price_min, price_max, n_bins + 1)

        # Accumulate volume in each bin
        volume_profile = []

        for i in range(len(bins) - 1):
            mask = (prices >= bins[i]) & (prices < bins[i + 1])
            bin_volume = volumes[mask].sum()
            bin_price = (bins[i] + bins[i + 1]) / 2

            if bin_volume > 0:
                volume_profile.append(
                    {
                        "price": float(bin_price),
                        "volume": float(bin_volume),
                        "pct_of_total": float(bin_volume / volumes.sum()),
                    }
                )

        # Sort by volume and return top levels
        volume_profile.sort(key=lambda x: x["volume"], reverse=True)
        return volume_profile[:5]  # Top 5 volume nodes

    def semantic_summary(self) -> str:
        """Generate one-line summary for logging."""
        ratio_str = f"{self.volume_ratio:.2f}x"
        trend_str = f"{self.volume_trend} volume"

        if self.volume_strength == "high" and self.buy_volume_pct > 0.6:
            return f"High volume ({ratio_str} avg) with strong buying pressure"
        elif self.volume_strength == "high" and self.sell_volume_pct > 0.6:
            return f"High volume ({ratio_str} avg) with strong selling pressure"
        elif self.volume_strength == "low":
            return f"Low volume ({ratio_str} avg) - {trend_str}"
        else:
            return f"Volume {ratio_str} average, {trend_str}, VWAP ${self.vwap:,.2f}"

    def to_agent_context(self) -> dict[str, Any]:
        """Format for agent consumption."""
        # Determine buy pressure
        if self.buy_volume_pct > 0.6:
            buy_pressure = "bullish"
        elif self.sell_volume_pct > 0.6:
            buy_pressure = "bearish"
        else:
            buy_pressure = "neutral"

        context = {
            "indicator": "volume",
            "values": {
                "vwap": round(self.vwap, 2),
                "volume_ratio": round(self.volume_ratio, 2),
                "current_volume": self.current_volume,
                "buy_pct": round(self.buy_volume_pct, 3),
                "sell_pct": round(self.sell_volume_pct, 3),
            },
            "volume_state": self.volume_strength,
            "volume_trend": self.volume_trend,
            "buy_pressure": buy_pressure,
            "price_correlation": round(self.price_volume_correlation, 2),
            "interpretation": self._generate_interpretation(),
        }

        if self.volume_nodes:
            context["key_levels"] = [
                {"price": node["price"], "strength": node["pct_of_total"]}
                for node in self.volume_nodes[:3]
            ]

        return context

    def _generate_interpretation(self) -> str:
        """Generate detailed interpretation for agents."""
        if self.volume_strength == "high":
            if self.buy_volume_pct > 0.7:
                return (
                    "Unusually high volume with strong buying pressure. "
                    "Potential breakout or accumulation phase. "
                    "Watch for continuation if price breaks resistance."
                )
            elif self.sell_volume_pct > 0.7:
                return (
                    "High volume selling detected. Distribution phase likely. "
                    "Risk of further downside if support breaks. "
                    "Consider defensive positioning."
                )
            else:
                return (
                    "High volume but mixed sentiment. "
                    "Market at decision point. "
                    "Wait for directional clarity."
                )
        elif self.volume_strength == "low":
            return (
                "Low volume indicates lack of conviction. "
                "Moves may not be sustainable. "
                "Wait for volume confirmation before trading."
            )
        else:
            if self.volume_trend == "increasing":
                return (
                    "Volume building suggests growing interest. "
                    "Monitor for potential breakout. "
                    f"VWAP at ${self.vwap:,.2f} acts as dynamic support/resistance."
                )
            else:
                return (
                    f"Normal volume conditions. VWAP at ${self.vwap:,.2f}. "
                    f"Buy pressure {self.buy_volume_pct:.0%} vs "
                    f"sell {self.sell_volume_pct:.0%}. "
                    "No significant volume signals."
                )

    def suggest_signal(self) -> dict[str, str]:
        """Suggest trading signal based on volume analysis."""
        # High volume scenarios
        if self.volume_strength == "high":
            if self.buy_volume_pct > 0.7:
                return {
                    "bias": "bullish",
                    "strength": "strong",
                    "reason": "High volume buying pressure detected",
                    "action": "consider_long"
                    if self.volume_trend == "increasing"
                    else "monitor",
                }
            elif self.sell_volume_pct > 0.7:
                return {
                    "bias": "bearish",
                    "strength": "strong",
                    "reason": "High volume selling pressure detected",
                    "action": "consider_short"
                    if self.volume_trend == "increasing"
                    else "reduce_risk",
                }

        # Low volume warning
        elif self.volume_strength == "low":
            return {
                "bias": "neutral",
                "strength": "weak",
                "reason": "Low volume - lack of market participation",
                "action": "wait",
            }

        # Moderate volume with trend
        elif self.volume_trend == "increasing" and self.buy_volume_pct > 0.6:
            return {
                "bias": "bullish",
                "strength": "moderate",
                "reason": "Increasing volume with buy pressure",
                "action": "prepare_entry",
            }
        elif self.volume_trend == "increasing" and self.sell_volume_pct > 0.6:
            return {
                "bias": "bearish",
                "strength": "moderate",
                "reason": "Increasing volume with sell pressure",
                "action": "consider_exit",
            }

        # Default neutral
        return {
            "bias": "neutral",
            "strength": "weak",
            "reason": "No clear volume signals",
            "action": "monitor",
        }
