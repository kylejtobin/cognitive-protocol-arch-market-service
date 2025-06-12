"""
Base classes and registry for technical indicators.

This module provides the abstract base class for all technical indicators
and a registry system for dynamic indicator loading.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound="BaseIndicator")


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
    def to_agent_context(self) -> dict[str, Any]:
        """
        Format indicator data for agent consumption.

        This should provide rich, structured data that agents can use for
        decision-making. The format should be consistent and include all
        relevant information.

        Returns:
            Dictionary with structured indicator data

        Example:
            {
                "indicator": "rsi",
                "value": 75.2,
                "state": "overbought",
                "signals": {
                    "divergence": "bearish",
                    "trend": "weakening"
                },
                "interpretation": "Price is overbought with bearish divergence..."
            }

        """
        ...

    @abstractmethod
    def suggest_signal(self) -> dict[str, str]:
        """
        Suggest a trading signal based on the indicator state.

        This provides actionable suggestions but does NOT make decisions.
        Agents are responsible for interpreting these suggestions.

        Returns:
            Dictionary with signal suggestions

        Example:
            {
                "bias": "bearish",
                "strength": "moderate",
                "reason": "RSI overbought with divergence",
                "action": "consider_reducing_position"
            }

        """
        ...


# IndicatorRegistry has been moved to registry.py
