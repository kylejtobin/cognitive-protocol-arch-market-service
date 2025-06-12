"""
Enums for market structure.

This module defines the standardized enum values used throughout the market data system.
These enums represent the semantic vocabulary of the domain and establish consistent
naming across exchanges and components.

"""

from __future__ import annotations

import enum

# =============================================================================
# MARKET STRUCTURE ENUMS
# =============================================================================


class Exchange(str, enum.Enum):
    """
    Supported exchange identifiers.

    These identifiers represent the exchanges integrated with the system
    and are used for routing, filtering, and identifying data sources.
    """

    COINBASE = "coinbase"
    BINANCE = "binance"
    BYBIT = "bybit"
    KRAKEN = "kraken"


class MessageType(str, enum.Enum):
    """
    Exchange-agnostic message type identifiers.

    These values categorize the type of market data message received
    from exchange websockets, regardless of the specific exchange format.
    """

    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADE = "trade"
    HEARTBEAT = "heartbeat"
    STATUS = "status"
    SUBSCRIPTION = "subscription"
    CANDLE = "candle"
    UNKNOWN = "unknown"


class OrderBookEventType(str, enum.Enum):
    """
    Types of order book events.

    These values distinguish between different types of order book updates,
    affecting how they should be processed and applied to the local state.
    """

    SNAPSHOT = "snapshot"  # Complete replacement of order book state
    UPDATE = "update"  # Incremental update to existing state
    DIFF = "diff"  # Differential update with specific changes
    RESET = "reset"  # Signal to reset and rebuild the order book


class OrderBookSide(str, enum.Enum):
    """
    Order book side identifiers.

    Represents the side of an order book (bid/buy or ask/sell).
    Used to categorize price levels and orders.
    """

    BID = "bid"  # Buy side (willing to buy at this price or lower)
    ASK = "ask"  # Sell side (willing to sell at this price or higher)


class TradeSide(str, enum.Enum):
    """
    Standardized enum for trade sides.

    Represents the direction of a trade (buy or sell) in a
    consistent format across all exchanges.
    """

    BUY = "buy"  # Trade executed as a buy
    SELL = "sell"  # Trade executed as a sell

    @classmethod
    def from_exchange(cls, side: str) -> TradeSide:
        """
        Convert exchange side format to standardized enum.

        Args:
            side: Exchange side string (e.g., "BUY", "SELL", "B", "S")

        Returns:
            Standardized TradeSide enum value

        """
        normalized = side.lower()
        if normalized in {"buy", "b", "bid"}:
            return cls.BUY
        elif normalized in {"sell", "s", "ask"}:
            return cls.SELL
        else:
            raise ValueError(f"Invalid trade side: {side}")


class MarketStatus(str, enum.Enum):
    """
    Exchange market status values.

    Represents the operational status of a market/product on an exchange,
    affecting whether trading operations can be performed.
    """

    OPEN = "open"  # Market is fully operational
    CLOSED = "closed"  # Market is not available for trading
    LIMITED = "limited"  # Market has limited functionality
    PENDING = "pending"  # Market is preparing to open
    HALTED = "halted"  # Market is temporarily suspended
    UNKNOWN = "unknown"  # Market status could not be determined
