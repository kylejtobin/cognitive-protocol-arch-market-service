"""
Order book state management with incremental update support.

This module handles the mutable order book state that Coinbase streams require.
It maintains sorted bid/ask levels and applies incremental updates efficiently.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from src.market.model.types import PriceLevelData, PriceSize


class PriceLevelSequence:
    """
    Sequence of price levels with protocol compliance.

    Wraps a list of PriceLevelData to provide the protocol interface.
    """

    def __init__(self, levels: list[PriceLevelData], reverse: bool = False) -> None:
        """
        Initialize with list of price level data.

        Args:
            levels: List of PriceLevelData
            reverse: If True, sort in descending order (for bids)

        """
        self._levels = sorted(levels, key=lambda x: x.price, reverse=reverse)

    def __len__(self) -> int:
        """Get the number of price levels."""
        return len(self._levels)

    def __getitem__(self, index: int) -> PriceSize:
        """Get a price level by index."""
        level = self._levels[index]
        return PriceSize(price=level.price, size=level.size)

    def __iter__(self) -> Sequence[PriceSize]:
        """Iterate through price levels."""
        return [PriceSize(price=level.price, size=level.size) for level in self._levels]

    @property
    def best_price(self) -> Decimal | None:
        """Get the best price."""
        return self._levels[0].price if self._levels else None

    def get_price_at_index(self, index: int) -> Decimal:
        """Get the price at a specific index."""
        return self._levels[index].price

    def get_size_at_index(self, index: int) -> Decimal:
        """Get the size at a specific index."""
        return self._levels[index].size

    def get_total_size(self) -> Decimal:
        """Get the total size of all price levels."""
        return Decimal(sum(level.size for level in self._levels))


class OrderBook(BaseModel):
    """
    Order book implementation with protocol compliance.

    This is a simple wrapper that provides the protocol interface over
    bid/ask level data. It's designed to be created fresh for each snapshot.
    """

    symbol: str
    timestamp: datetime
    bid_levels: list[PriceLevelData]
    ask_levels: list[PriceLevelData]
    sequence: int | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def bids(self) -> PriceLevelSequence:
        """Get the bid levels."""
        return PriceLevelSequence(self.bid_levels, reverse=True)

    @property
    def asks(self) -> PriceLevelSequence:
        """Get the ask levels."""
        return PriceLevelSequence(self.ask_levels, reverse=False)

    @property
    def best_bid(self) -> Decimal | None:
        """Get the best bid price."""
        return self.bid_levels[0].price if self.bid_levels else None

    @property
    def best_ask(self) -> Decimal | None:
        """Get the best ask price."""
        return self.ask_levels[0].price if self.ask_levels else None

    @property
    def mid_price(self) -> Decimal | None:
        """Get the mid price."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> Decimal | None:
        """Get the spread."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def spread_percentage(self) -> Decimal | None:
        """Get the spread percentage."""
        if self.spread is None or self.mid_price is None or self.mid_price == 0:
            return None
        return (self.spread / self.mid_price) * 100


class MutableOrderBook:
    """
    Mutable order book state for maintaining Coinbase level2 data.

    This class handles the incremental updates from Coinbase WebSocket
    and maintains the current book state. It can produce immutable
    OrderBook snapshots for the protocol.
    """

    def __init__(self, symbol: str) -> None:
        """Initialize the mutable order book."""
        self.symbol = symbol
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}
        self.sequence: int | None = None
        self.last_update = datetime.now(UTC)

    def apply_snapshot(
        self,
        bids: list[PriceLevelData],
        asks: list[PriceLevelData],
        sequence: int | None = None,
    ) -> None:
        """Replace entire book with snapshot data."""
        self.bids.clear()
        self.asks.clear()

        for level in bids:
            if level.size > 0:
                self.bids[level.price] = level.size

        for level in asks:
            if level.size > 0:
                self.asks[level.price] = level.size

        self.sequence = sequence
        self.last_update = datetime.now(UTC)

    def apply_update(self, side: str, price: Decimal, size: Decimal) -> None:
        """Apply incremental update to a price level."""
        book = self.bids if side == "bid" else self.asks

        if size == 0:
            # Remove price level
            book.pop(price, None)
        else:
            # Update price level
            book[price] = size

        self.last_update = datetime.now(UTC)

    def to_protocol(self) -> OrderBook:
        """Create immutable protocol-compliant snapshot."""
        # Sort bids descending, asks ascending
        bid_items = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
        ask_items = sorted(self.asks.items(), key=lambda x: x[0])

        # Convert to PriceLevelData using Pydantic validation
        bid_levels = [PriceLevelData(price=p, size=s) for p, s in bid_items]
        ask_levels = [PriceLevelData(price=p, size=s) for p, s in ask_items]

        return OrderBook(
            symbol=self.symbol,
            timestamp=self.last_update,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            sequence=self.sequence,
        )
