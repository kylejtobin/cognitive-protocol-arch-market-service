"""Market analysis components."""

from src.market.analysis.base import BaseIndicator
from src.market.analysis.registry import IndicatorRegistry, register

__all__ = [
    "BaseIndicator",
    "IndicatorRegistry",
    "register",
]


def __getattr__(name: str) -> type[BaseIndicator]:
    """
    Lazy load indicator classes.

    This allows importing indicators without circular dependencies.
    """
    if name == "RSIAnalysis":
        from src.market.analysis.momentum_indicators import RSIAnalysis

        return RSIAnalysis
    elif name == "MACDAnalysis":
        from src.market.analysis.macd import MACDAnalysis

        return MACDAnalysis
    elif name == "StochasticAnalysis":
        from src.market.analysis.stochastic import StochasticAnalysis

        return StochasticAnalysis
    elif name == "VolumeAnalysis":
        from src.market.analysis.volume_analysis import VolumeAnalysis

        return VolumeAnalysis
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Force registration of all indicators
from . import macd, momentum_indicators, stochastic, volume_analysis  # noqa: F401, E402
