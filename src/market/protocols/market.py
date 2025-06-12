"""
Market Protocol Layer for Market Data Service.

This module defines the core domain protocols for market data entities. These protocols
represent the semantic contracts that domain objects must fulfill, emphasizing
what market data means rather than how it's represented or transported.

Key design principles:
- Semantic clarity: Protocols focus on meaning and relationships
- Layered isolation: Clear separation from exchange-specific concepts
- Precise type signatures: Using appropriate types for financial data
- Nullable properties: Optional fields may return None when unavailable
- Temporal guarantees: Consistent handling of time-based relationships

The market layer provides:
1. Semantically rich abstractions of market concepts
2. Protocol verification for domain objects
3. Consistent behavioral contracts
4. Type safety for market operations
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Protocol, runtime_checkable

from src.market.enums import MarketStatus, TradeSide

# Type aliases for simple raw types
RawPrice = Decimal | float | str
RawSize = Decimal | float | str


# =============================================================================
# COLLECTION PROTOCOLS
# =============================================================================


@runtime_checkable
class MarketPriceLevelSequenceProtocol(Protocol):
    """
    Protocol for sequence of price levels.

    Semantic Role: Ordered liquidity representation
    Relationships:
    - Component of: MarketOrderBookProtocol (as bids or asks)
    - Contains: MarketPriceLevelProtocol instances
    - Semantic Guarantees: Correct price ordering and uniqueness

    Represents one side (bid or ask) of the order book with
    proper price ordering and level uniqueness guarantees.
    """

    def __len__(self) -> int:
        """
        Get number of price levels.

        Semantic Role: Depth measurement
        Relationships:
        - Indicator: Market liquidity breadth
        - Performance: Used for optimization decisions

        Returns:
            Number of price levels in sequence

        """
        ...

    def __getitem__(self, index: int) -> MarketPriceLevelProtocol:
        """
        Get price level at index.

        Semantic Role: Position-based access
        Relationships:
        - Ordering: Index 0 is always best price
        - Constraint: Raises error for invalid index

        Args:
            index: Position in the sequence

        Returns:
            Price level at the specified index

        """
        ...

    def __iter__(self) -> Sequence[MarketPriceLevelProtocol]:
        """
        Iterate through price levels.

        Semantic Role: Sequential access pattern
        Relationships:
        - Ordering: Maintains correct price ordering
        - Performance: Efficient for processing all levels

        Returns:
            Iterator over price levels

        """
        ...

    @property
    def best_price(self) -> Decimal | None:
        """
        Get best price in this sequence.

        Semantic Role: Market best quote
        Relationships:
        - For bids: Highest buy price
        - For asks: Lowest sell price
        - Empty check: None if no levels exist

        Returns:
            Best price or None if empty

        """
        ...

    def get_price_at_index(self, index: int) -> Decimal:
        """
        Get price at specific index.

        Semantic Role: Direct price access
        Relationships:
        - Ordering: Index 0 is always best price
        - Shortcut: For price-only access without level object

        Args:
            index: Position in the sequence

        Returns:
            Price at the specified index

        """
        ...

    def get_size_at_index(self, index: int) -> Decimal:
        """
        Get size at specific index.

        Semantic Role: Direct size access
        Relationships:
        - Ordering: Index 0 is always best price
        - Shortcut: For size-only access without level object

        Args:
            index: Position in the sequence

        Returns:
            Size at the specified index

        """
        ...

    def get_total_size(self) -> Decimal:
        """
        Get total size across all levels.

        Semantic Role: Aggregate liquidity measure
        Relationships:
        - Aggregation: Sum of sizes across all levels
        - Indicator: Total available liquidity on this side

        Returns:
            Total size as Decimal

        """
        ...


# =============================================================================
# DOMAIN PRIMITIVE PROTOCOLS
# =============================================================================


@runtime_checkable
class MarketCurrencyProtocol(Protocol):
    """
    Protocol for currency representation.

    Semantic Role: Fundamental asset identifier
    Relationships:
    - Component of: MarketSymbolProtocol (as base or quote)
    - Referenced by: Trading pairs, balances, and pricing
    - Semantic Guarantees: Unique identification and precision control

    Currencies are the atomic units of trade and value representation.
    They maintain precision information for accurate value calculations.
    """

    @property
    def code(self) -> str:
        """
        Get currency code.

        Semantic Role: Primary identifier
        Relationships:
        - Uniqueness: Must be unique within trading venue
        - Format: Typically 3-5 character code (e.g., BTC, ETH, USD)

        Returns:
            Standardized currency code

        """
        ...

    @property
    def name(self) -> str:
        """
        Get full currency name.

        Semantic Role: Human-readable identifier
        Relationships:
        - Display: Used in user interfaces
        - Descriptive: Provides context beyond code

        Returns:
            Full name of the currency

        """
        ...

    @property
    def precision(self) -> int:
        """
        Get decimal precision.

        Semantic Role: Value representation constraint
        Relationships:
        - Validation: Maximum allowed decimal places
        - Formatting: How values should be displayed

        Returns:
            Number of decimal places for this currency

        """
        ...

    def __eq__(self, other: object) -> bool:
        """
        Compare currencies for equality.

        Semantic Role: Identity comparison
        Relationships:
        - Based on: Currency code only (canonical identifier)
        - Ignores: Presentation details like name

        Args:
            other: Another currency to compare with

        Returns:
            True if currencies have the same code

        """
        ...


@runtime_checkable
class MarketSymbolProtocol(Protocol):
    """
    Protocol for market symbol representation.

    Semantic Role: Trading pair identifier
    Relationships:
    - References: Base and quote currencies
    - Referenced by: All market data entities
    - Semantic Guarantees: Consistent trading pair identification

    Represents a trading pair with base and quote currencies,
    providing the primary key for market data organization.
    """

    @property
    def symbol(self) -> str:
        """
        Get market identifier.

        Semantic Role: Primary market identifier
        Relationships:
        - Uniqueness: Must be unique within trading venue
        - Format: Base-Quote format (e.g., BTC-USD)

        Returns:
            Standardized market symbol string

        """
        ...

    @property
    def base_currency(self) -> MarketCurrencyProtocol:
        """
        Get base currency.

        Semantic Role: Asset being traded
        Relationships:
        - First part: First currency in symbol
        - Example: BTC in BTC-USD

        Returns:
            Base currency object

        """
        ...

    @property
    def quote_currency(self) -> MarketCurrencyProtocol:
        """
        Get quote currency.

        Semantic Role: Pricing currency
        Relationships:
        - Second part: Second currency in symbol
        - Example: USD in BTC-USD

        Returns:
            Quote currency object

        """
        ...

    @property
    def min_size(self) -> Decimal:
        """
        Get minimum order size.

        Semantic Role: Order size constraint
        Relationships:
        - Validation: Minimum allowed order size
        - Exchange rule: Varies by trading venue

        Returns:
            Minimum order size

        """
        ...

    @property
    def price_increment(self) -> Decimal:
        """
        Get price tick size.

        Semantic Role: Price granularity specification
        Relationships:
        - Validation: Smallest allowed price movement
        - Exchange rule: Varies by trading venue

        Returns:
            Price increment

        """
        ...

    def __eq__(self, other: object) -> bool:
        """
        Compare symbols for equality.

        Semantic Role: Identity comparison
        Relationships:
        - Based on: Symbol string only (canonical identifier)
        - Ignores: Exchange-specific details

        Args:
            other: Another symbol to compare with

        Returns:
            True if symbols have the same identifier

        """
        ...


@runtime_checkable
class MarketPriceProtocol(Protocol):
    """
    Protocol for price values.

    Semantic Role: Financial value with decimal precision
    Relationships:
    - Used by: Orders, trades, price levels
    - Constraints: Non-negative, decimal precision
    - Semantic Guarantees: Preserves exact decimal value without floating point errors
    """

    @property
    def value(self) -> Decimal:
        """Get the decimal value with full precision."""
        ...

    @property
    def as_float(self) -> float:
        """Get the value as a float (with acknowledged precision loss)."""
        ...


@runtime_checkable
class MarketSizeProtocol(Protocol):
    """
    Protocol for size/quantity values.

    Semantic Role: Asset quantity with decimal precision
    Relationships:
    - Used by: Orders, trades, positions
    - Constraints: Non-negative, decimal precision
    - Semantic Guarantees: Preserves exact decimal value without floating point errors
    """

    @property
    def value(self) -> Decimal:
        """Get the decimal value with full precision."""
        ...

    @property
    def as_float(self) -> float:
        """Get the value as a float (with acknowledged precision loss)."""
        ...


@runtime_checkable
class MarketTimestampProtocol(Protocol):
    """
    Protocol for market timestamp representation.

    Semantic Role: Temporal context for market events
    Relationships:
    - Component of: All time-sensitive market data
    - Semantic Guarantees: Consistent time representation

    Provides temporal context for market data,
    enabling sequencing and freshness determination.
    """

    @property
    def timestamp(self) -> datetime:
        """
        Get timestamp value.

        Semantic Role: Point in time
        Relationships:
        - Format: UTC datetime
        - Precision: Millisecond or better

        Returns:
            UTC datetime

        """
        ...

    @property
    def iso_format(self) -> str:
        """
        Get ISO 8601 formatted timestamp.

        Semantic Role: Standardized time representation
        Relationships:
        - Format: ISO 8601 with timezone
        - Used for: Serialization and logging

        Returns:
            ISO 8601 formatted timestamp string

        """
        ...

    @property
    def epoch_ms(self) -> int:
        """
        Get millisecond epoch timestamp.

        Semantic Role: Numeric time representation
        Relationships:
        - Format: Milliseconds since Unix epoch
        - Used for: Time calculations and comparisons

        Returns:
            Millisecond epoch timestamp

        """
        ...

    def __eq__(self, other: object) -> bool:
        """
        Compare timestamps for equality.

        Semantic Role: Temporal equality
        Relationships:
        - Precision: Millisecond level comparison
        - Ignores: Representation differences

        Args:
            other: Another timestamp to compare with

        Returns:
            True if timestamps represent same point in time

        """
        ...

    def __lt__(self, other: object) -> bool:
        """
        Compare timestamps for ordering.

        Semantic Role: Temporal ordering
        Relationships:
        - Enables: Chronological sorting
        - Used for: Sequencing market events

        Args:
            other: Another timestamp to compare with

        Returns:
            True if this timestamp is earlier than other

        """
        ...


# =============================================================================
# MARKET DATA PROTOCOLS
# =============================================================================
@runtime_checkable
class MarketHeartbeatProtocol(Protocol):
    """
    Protocol for exchange heartbeat messages.

    Semantic Role: Connection health signal
    Relationships:
    - Emitted by: Exchange WebSocket connections
    - Consumed by: Service components managing connection state
    - Semantic Guarantees: Sequence ordering and timestamp accuracy
    """

    @property
    def sequence(self) -> int:
        """Sequence number of this heartbeat."""
        ...

    @property
    def timestamp(self) -> MarketTimestampProtocol:
        """Timestamp when this heartbeat was generated."""
        ...


@runtime_checkable
class MarketPriceLevelProtocol(Protocol):
    """
    Protocol for price level in order book.

    Semantic Role: Single liquidity point in order book
    Relationships:
    - Component of: MarketOrderBookProtocol
    - Composed of: Price and size at specific level
    - Semantic Guarantees: Price uniqueness within side

    Price levels represent available liquidity at a specific price
    point, with aggregated size from potentially multiple orders.
    """

    @property
    def price(self) -> Decimal:
        """
        Get price of this level.

        Semantic Role: Price point identifier
        Relationships:
        - Ordering: Determines position in order book
        - Uniqueness: Must be unique within a side

        Returns:
            Price as Decimal with full precision

        """
        ...

    @property
    def size(self) -> Decimal:
        """
        Get aggregated size at this price.

        Semantic Role: Available liquidity measure
        Relationships:
        - Aggregation: Sum of all orders at this price
        - Constraint: Always positive, zero means level removal

        Returns:
            Size as Decimal with full precision

        """
        ...


@runtime_checkable
class MarketTickerProtocol(Protocol):
    """
    Protocol for market ticker data.

    Semantic Role: Market summary information
    Relationships:
    - Component of: MarketSnapshotProtocol
    - Summarizes: Recent market activity
    - Semantic Guarantees: Consistent price and volume information

    Provides a concise summary of market state including latest price,
    best quotes, and trading volume information.
    """

    @property
    def symbol(self) -> str:
        """
        Get market identifier.

        Semantic Role: Market context
        Relationships:
        - Reference: Trading pair this ticker represents
        - Uniqueness: Identifies specific market

        Returns:
            Market symbol string

        """
        ...

    @property
    def price(self) -> Decimal:
        """
        Get latest price.

        Semantic Role: Current market price
        Relationships:
        - Derived from: Most recent trade(s)
        - Used for: Price display and calculations

        Returns:
            Latest traded price

        """
        ...

    @property
    def bid(self) -> Decimal | None:
        """
        Get best bid price.

        Semantic Role: Best buy price available
        Relationships:
        - Optional: May be None if unavailable
        - Corresponds to: Top of bid side in order book

        Returns:
            Best bid price or None if unavailable

        """
        ...

    @property
    def ask(self) -> Decimal | None:
        """
        Get best ask price.

        Semantic Role: Best sell price available
        Relationships:
        - Optional: May be None if unavailable
        - Corresponds to: Top of ask side in order book

        Returns:
            Best ask price or None if unavailable

        """
        ...

    @property
    def volume(self) -> Decimal | None:
        """
        Get 24-hour trading volume.

        Semantic Role: Trading activity measure
        Relationships:
        - Optional: May be None if unavailable
        - Timeframe: Typically represents 24-hour period

        Returns:
            24-hour volume or None if unavailable

        """
        ...

    @property
    def timestamp(self) -> datetime:
        """
        Get ticker timestamp.

        Semantic Role: Temporal context
        Relationships:
        - Validation: When this information was known to be valid
        - Freshness: Indicates data recency

        Returns:
            UTC datetime of ticker creation

        """
        ...


@runtime_checkable
class MarketTradeProtocol(Protocol):
    """
    Protocol for market trade execution.

    Semantic Role: Individual trade execution record
    Relationships:
    - Component of: MarketSnapshotProtocol (as part of trade sequence)
    - Represents: Completed market transaction
    - Semantic Guarantees: Complete trade execution details

    Represents a single completed trade with price, size, side,
    and temporal information for analysis and display.
    """

    @property
    def symbol(self) -> str:
        """
        Get market identifier.

        Semantic Role: Market context
        Relationships:
        - Reference: Trading pair where trade occurred
        - Uniqueness: Identifies specific market

        Returns:
            Market symbol string

        """
        ...

    @property
    def price(self) -> Decimal:
        """
        Get trade price.

        Semantic Role: Transaction clearing price
        Relationships:
        - Represents: Agreed price between buyer and seller
        - Used for: Price discovery and chart data

        Returns:
            Executed trade price

        """
        ...

    @property
    def size(self) -> Decimal:
        """
        Get trade size.

        Semantic Role: Transaction volume
        Relationships:
        - Represents: Quantity exchanged between buyer and seller
        - Used for: Volume indicators and trade significance

        Returns:
            Executed trade size

        """
        ...

    @property
    def side(self) -> TradeSide:
        """
        Get trade aggressor side.

        Semantic Role: Trade direction classifier
        Relationships:
        - Indicates: Whether taker was buyer or seller
        - Used for: Trade direction indicators

        Returns:
            Trade side enum (buy or sell)

        """
        ...

    @property
    def timestamp(self) -> datetime:
        """
        Get trade timestamp.

        Semantic Role: Temporal context
        Relationships:
        - Indicates: When trade execution occurred
        - Used for: Time-sequenced analysis of trades

        Returns:
            UTC datetime of trade execution

        """
        ...

    @property
    def trade_id(self) -> str | None:
        """
        Get unique trade identifier.

        Semantic Role: Trade uniqueness guarantee
        Relationships:
        - Optional: May be None if unavailable
        - Used for: Deduplication and reference

        Returns:
            Trade ID or None if unavailable

        """
        ...


@runtime_checkable
class MarketOrderBookProtocol(Protocol):
    """
    Protocol for order book representation.

    Semantic Role: Complete market depth snapshot
    Relationships:
    - Contains: Bid and ask MarketPriceLevelSequenceProtocol
    - Component of: MarketSnapshotProtocol
    - Semantic Guarantees: Consistent price ordering, no empty levels

    Represents the full depth of market with buy and sell sides,
    maintaining price ordering and level consistency guarantees.
    """

    @property
    def symbol(self) -> str:
        """
        Get market identifier.

        Semantic Role: Market context
        Relationships:
        - Reference: Trading pair this book belongs to
        - Uniqueness: Identifies specific market

        Returns:
            Market symbol string

        """
        ...

    @property
    def bids(self) -> MarketPriceLevelSequenceProtocol:
        """
        Get buy side of book.

        Semantic Role: Buy liquidity representation
        Relationships:
        - Ordering: Descending price order (best bid first)
        - Complete: Contains all known bid levels

        Returns:
            Sequence of bid levels

        """
        ...

    @property
    def asks(self) -> MarketPriceLevelSequenceProtocol:
        """
        Get sell side of book.

        Semantic Role: Sell liquidity representation
        Relationships:
        - Ordering: Ascending price order (best ask first)
        - Complete: Contains all known ask levels

        Returns:
            Sequence of ask levels

        """
        ...

    @property
    def best_bid(self) -> Decimal | None:
        """
        Get best bid price.

        Semantic Role: Best buy price available
        Relationships:
        - Shortcut: Convenience for bids.best_price
        - Empty check: None if no bids exist

        Returns:
            Best bid price or None if empty

        """
        ...

    @property
    def best_ask(self) -> Decimal | None:
        """
        Get best ask price.

        Semantic Role: Best sell price available
        Relationships:
        - Shortcut: Convenience for asks.best_price
        - Empty check: None if no asks exist

        Returns:
            Best ask price or None if empty

        """
        ...

    @property
    def mid_price(self) -> Decimal | None:
        """
        Get mid price between best bid and ask.

        Semantic Role: Fair value estimate
        Relationships:
        - Derived: (best_bid + best_ask) / 2
        - Empty check: None if either side is empty

        Returns:
            Mid price or None if insufficient data

        """
        ...

    @property
    def spread(self) -> Decimal | None:
        """
        Get spread between best bid and ask.

        Semantic Role: Market tightness measure
        Relationships:
        - Derived: best_ask - best_bid
        - Empty check: None if either side is empty

        Returns:
            Spread or None if insufficient data

        """
        ...

    @property
    def spread_percentage(self) -> Decimal | None:
        """
        Get spread as percentage of mid price.

        Semantic Role: Relative market tightness measure
        Relationships:
        - Derived: (spread / mid_price) * 100
        - Empty check: None if either side is empty

        Returns:
            Spread percentage or None if insufficient data

        """
        ...

    @property
    def timestamp(self) -> datetime:
        """
        Get order book timestamp.

        Semantic Role: Temporal context
        Relationships:
        - Validation: When this state was known to be valid
        - Ordering: Allows sequencing of multiple snapshots

        Returns:
            UTC datetime of snapshot creation

        """
        ...

    @property
    def sequence(self) -> int | None:
        """
        Get order book sequence number.

        Semantic Role: Consistency validation token
        Relationships:
        - Optional: May be None if exchange doesn't provide
        - Ordering: Monotonically increasing for updates

        Returns:
            Sequence number or None if unavailable

        """
        ...


@runtime_checkable
class MarketStatusProtocol(Protocol):
    """
    Protocol for market operational status.

    Semantic Role: Market operational state information
    Relationships:
    - Component of: MarketSnapshotProtocol
    - Represents: Current trading status
    - Semantic Guarantees: Clear operational state classification

    Provides information about market operational status,
    indicating whether trading is available or suspended.
    """

    @property
    def symbol(self) -> str:
        """
        Get market identifier.

        Semantic Role: Market context
        Relationships:
        - Reference: Trading pair this status applies to
        - Uniqueness: Identifies specific market

        Returns:
            Market symbol string

        """
        ...

    @property
    def status(self) -> MarketStatus:
        """
        Get market status.

        Semantic Role: Market operational state
        Relationships:
        - Indicates: Whether market is open, closed, or in special state
        - Used for: Trading availability determination

        Returns:
            Market status enum

        """
        ...

    @property
    def timestamp(self) -> datetime:
        """
        Get status update timestamp.

        Semantic Role: Temporal context
        Relationships:
        - Indicates: When status change occurred
        - Used for: State transition tracking

        Returns:
            UTC datetime of status update

        """
        ...

    @property
    def reason(self) -> str | None:
        """
        Get reason for status change.

        Semantic Role: Status change explanation
        Relationships:
        - Optional: May be None if unavailable
        - Used for: User notifications and logging

        Returns:
            Status change reason or None if unavailable

        """
        ...


@runtime_checkable
class MarketSnapshotProtocol(Protocol):
    """
    Protocol for complete market state snapshot.

    Semantic Role: Complete market state representation
    Relationships:
    - Aggregates: All market data components
    - Semantic Guarantees: Temporal consistency and completeness

    Provides a comprehensive view of market state at a point in time,
    including order book, ticker, trades, and status information.
    """

    @property
    def symbol(self) -> str:
        """
        Get market identifier.

        Semantic Role: Market context
        Relationships:
        - Reference: Trading pair this snapshot represents
        - Uniqueness: Identifies specific market

        Returns:
            Market symbol string

        """
        ...

    @property
    def timestamp(self) -> datetime:
        """
        Get snapshot timestamp.

        Semantic Role: Temporal context
        Relationships:
        - Validation: When this state was known to be valid
        - Freshness: Indicates data recency

        Returns:
            UTC datetime of snapshot creation

        """
        ...

    @property
    def order_book(self) -> MarketOrderBookProtocol | None:
        """
        Get order book state.

        Semantic Role: Market depth information
        Relationships:
        - Optional: May be None if unavailable
        - Complete: Full order book at snapshot time

        Returns:
            Order book or None if unavailable

        """
        ...

    @property
    def ticker(self) -> MarketTickerProtocol | None:
        """
        Get ticker state.

        Semantic Role: Market summary information
        Relationships:
        - Optional: May be None if unavailable
        - Summary: Concise market state at snapshot time

        Returns:
            Ticker or None if unavailable

        """
        ...

    @property
    def trades(self) -> Sequence[MarketTradeProtocol] | None:
        """
        Get recent trades.

        Semantic Role: Recent market activity
        Relationships:
        - Optional: May be None if unavailable
        - Ordering: Most recent trade first

        Returns:
            Sequence of trades or None if unavailable

        """
        ...

    @property
    def status(self) -> MarketStatusProtocol | None:
        """
        Get market status.

        Semantic Role: Market operational state
        Relationships:
        - Optional: May be None if unavailable
        - Valid: Current status at snapshot time

        Returns:
            Market status or None if unavailable

        """
        ...
