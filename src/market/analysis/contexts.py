"""
Typed agent context models for technical indicators.

These models provide type-safe representations of indicator data
for agent/AI consumption, replacing untyped dict[str, Any] returns.
"""

from typing import Literal

from pydantic import BaseModel, Field


class BaseAgentContext(BaseModel):
    """Base context that all indicators provide."""

    indicator: str = Field(description="Indicator name")
    interpretation: str = Field(description="Human-readable interpretation")


class SignalSuggestion(BaseModel):
    """Typed signal suggestion."""

    bias: Literal["bullish", "bearish", "neutral"]
    strength: Literal["strong", "moderate", "weak"]
    reason: str
    action: str


class RSIKeyLevels(BaseModel):
    """RSI key level information."""

    overbought: int = 70
    oversold: int = 30
    neutral: int = 50
    current: float


class RSISignals(BaseModel):
    """RSI signal information."""

    is_overbought: bool
    is_oversold: bool
    divergence: Literal["bullish", "bearish", "none"] | None = None


class RSIAgentContext(BaseAgentContext):
    """Type-safe context for RSI indicator."""

    indicator: Literal["rsi"] = "rsi"
    value: float
    state: Literal[
        "strongly_bullish", "bullish", "neutral", "bearish", "strongly_bearish"
    ]
    strength: Literal["extreme", "strong", "moderate", "weak"]
    key_levels: RSIKeyLevels
    signals: RSISignals


class MACDSignals(BaseModel):
    """MACD signal information."""

    histogram_trend: Literal["expanding", "contracting", "neutral"]
    signal_position: Literal["above_signal", "below_signal", "at_signal"]
    zero_cross: Literal["bullish", "bearish", "none"]


class MACDValues(BaseModel):
    """MACD calculation values."""

    macd: float
    signal: float
    histogram: float


class MACDAgentContext(BaseAgentContext):
    """Type-safe context for MACD indicator."""

    indicator: Literal["macd"] = "macd"
    values: MACDValues
    trend: Literal["bullish", "bearish", "neutral"]
    momentum: Literal["increasing", "decreasing", "stable"]
    signals: MACDSignals


class StochasticValues(BaseModel):
    """Stochastic oscillator values."""

    k_percent: float
    d_percent: float | None = None


class StochasticZones(BaseModel):
    """Stochastic zone information."""

    current_zone: Literal["overbought", "neutral", "oversold"]
    k_zone: Literal["overbought", "neutral", "oversold"]
    d_zone: Literal["overbought", "neutral", "oversold"] | None = None


class StochasticAgentContext(BaseAgentContext):
    """Type-safe context for Stochastic indicator."""

    indicator: Literal["stochastic"] = "stochastic"
    values: StochasticValues
    zones: StochasticZones
    momentum: Literal["bullish", "bearish", "neutral"]
    crossover: Literal["bullish", "bearish", "none"]


class VolumeProfile(BaseModel):
    """Volume profile information."""

    current_volume: float
    average_volume: float
    volume_ratio: float
    volume_trend: Literal["increasing", "decreasing", "stable"]


class VolumeSignals(BaseModel):
    """Volume-based signals."""

    is_high_volume: bool
    is_low_volume: bool
    volume_spike: bool
    volume_dry_up: bool


class VolumeAgentContext(BaseAgentContext):
    """Type-safe context for Volume Analysis."""

    indicator: Literal["volume"] = "volume"
    profile: VolumeProfile
    pressure: Literal["buying", "selling", "neutral"]
    strength: Literal["strong", "moderate", "weak"]
    signals: VolumeSignals


class MomentumValues(BaseModel):
    """Generic momentum indicator values."""

    current: float
    previous: float | None = None
    change: float | None = None


class MomentumAgentContext(BaseAgentContext):
    """Type-safe context for generic momentum indicators."""

    indicator: Literal["momentum"] = "momentum"
    values: MomentumValues
    state: Literal["accelerating", "decelerating", "stable"]
    direction: Literal["up", "down", "sideways"]
    strength: Literal["strong", "moderate", "weak"]


# Microstructure Analysis Contexts


class SpreadInfo(BaseModel):
    """Spread information for microstructure."""

    current_bps: float
    is_widening: bool
    volatility: float


class TradeFlowInfo(BaseModel):
    """Trade flow information for microstructure."""

    momentum_score: float
    buy_pressure: float
    trend: Literal["up", "down", "neutral"]
    volume: float


class MicrostructureAgentContext(BaseModel):
    """Type-safe context for market microstructure analysis."""

    symbol: str
    timestamp: str
    market_quality: Literal["excellent", "good", "fair", "poor"]
    liquidity_score: float
    execution_cost_bps: float
    spread: SpreadInfo
    trade_flow: TradeFlowInfo
    summary: str
