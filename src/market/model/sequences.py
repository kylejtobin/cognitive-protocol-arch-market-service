"""
Simple price level sequence implementation.

This module provides a lightweight implementation of MarketPriceLevelSequenceProtocol
that can be used by exchange adapters without complex dependencies.
"""

from collections.abc import Iterator
from decimal import Decimal


class SimplePriceLevel:
    """Satisfies MarketPriceLevelProtocol through structural typing."""

    def __init__(self, price: Decimal, size: Decimal) -> None:
        """Initialize the price level."""
        self._price = price
        self._size = size

    @property
    def price(self) -> Decimal:
        """Get the price."""
        return self._price

    @property
    def size(self) -> Decimal:
        """Get the size."""
        return self._size


class SimplePriceLevelSequence:
    """
    Simple implementation of price level sequence for protocol compliance.

    This is a lightweight alternative to PriceLevelSequence that works
    directly with tuples of (price, size) for efficiency.
    """

    def __init__(self, levels: list[tuple[Decimal, Decimal]]) -> None:
        """Initialize the price level sequence."""
        self._levels = levels

    def __len__(self) -> int:
        """Get the number of price levels."""
        return len(self._levels)

    def __getitem__(self, index: int) -> SimplePriceLevel:
        """Get a price level by index."""
        price, size = self._levels[index]
        return SimplePriceLevel(price, size)

    def __iter__(self) -> Iterator[SimplePriceLevel]:
        """Return iterator of price levels."""
        for price, size in self._levels:
            yield SimplePriceLevel(price, size)

    @property
    def best_price(self) -> Decimal | None:
        """Get the best price."""
        return self._levels[0][0] if self._levels else None

    def get_price_at_index(self, index: int) -> Decimal:
        """Get the price at a specific index."""
        return self._levels[index][0]

    def get_size_at_index(self, index: int) -> Decimal:
        """Get the size at a specific index."""
        return self._levels[index][1]

    def get_total_size(self) -> Decimal:
        """Get the total size of all price levels."""
        return Decimal(sum(size for _, size in self._levels))
