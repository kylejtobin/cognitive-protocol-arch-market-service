"""
Market Domain Layer.

This package contains domain primitives that provide semantic meaning
and rich behavior to market data values. These primitives work in
harmony with our protocol layer's neutral types.

Key principles:
- Protocols use neutral types (Decimal, str, datetime)
- Domain models use primitives internally for rich behavior
- Primitives can convert to neutral types for protocol satisfaction
"""

from src.market.domain.primitives import (
    VWAP,
    BasisPoints,
    Percentage,
    Price,
    Size,
    Spread,
    Volume,
)

__all__ = [
    "VWAP",
    "BasisPoints",
    "Percentage",
    "Price",
    "Size",
    "Spread",
    "Volume",
]
