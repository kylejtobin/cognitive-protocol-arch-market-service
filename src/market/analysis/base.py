"""
Base classes and registry for technical indicators.

This module provides the abstract base class for all technical indicators
and a registry system for dynamic indicator loading.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from src.market.analysis.contexts import (
    BaseAgentContext,
    MACDAgentContext,
    MomentumAgentContext,
    RSIAgentContext,
    SignalSuggestion,
    StochasticAgentContext,
    VolumeAgentContext,
)

T = TypeVar("T", bound="BaseIndicator")

# Union type for all possible agent contexts
AgentContext = (
    RSIAgentContext
    | MACDAgentContext
    | StochasticAgentContext
    | VolumeAgentContext
    | MomentumAgentContext
    | BaseAgentContext  # Fallback for custom indicators
)


class BaseIndicator(BaseModel, ABC):
    """
    Abstract base class for all technical indicators.

    All indicators must inherit from this class and implement the required methods.
    This ensures consistency across indicators and standardized output for agents.
    """

    model_config = ConfigDict(frozen=True)

    # Required fields for all indicators
    symbol: str = Field(description="Trading symbol")
    timestamp: datetime = Field(description="Analysis timestamp")

    @abstractmethod
    def semantic_summary(self) -> str:
        """
        Provide a one-line semantic summary of the indicator state.

        This should be human-readable and suitable for logging or quick status checks.

        Returns:
            One-line summary string

        Example:
            "RSI: 75.2 (overbought, bearish divergence detected)"

        """
        ...

    @abstractmethod
    def to_agent_context(self) -> AgentContext:
        """
        Format indicator data for agent consumption.

        This returns a typed Pydantic model that agents can use for
        decision-making. The model provides full type safety and validation.

        Returns:
            Typed agent context (specific to each indicator)

        Example:
            return RSIAgentContext(
                value=75.2,
                state="overbought",
                strength="strong",
                key_levels=RSIKeyLevels(current=75.2),
                signals=RSISignals(is_overbought=True, is_oversold=False),
                interpretation="RSI indicates overbought conditions..."
            )

        """
        ...

    @abstractmethod
    def suggest_signal(self) -> SignalSuggestion:
        """
        Suggest a trading signal based on the indicator state.

        This provides actionable suggestions but does NOT make decisions.
        Agents are responsible for interpreting these suggestions.

        Returns:
            Typed signal suggestion

        Example:
            return SignalSuggestion(
                bias="bearish",
                strength="moderate",
                reason="RSI overbought with divergence",
                action="consider_reducing_position"
            )

        """
        ...


# IndicatorRegistry has been moved to registry.py
