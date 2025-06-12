"""
Coinbase WebSocket API Pydantic Models.

This module implements Pydantic models that parse Coinbase WebSocket messages
and expose protocol-compliant properties. The models satisfy market protocols
through structural typing (duck typing) without explicitly inheriting from them.

Key design principles:
- Pydantic models inherit ONLY from BaseModel
- Raw fields store exchange data as-is (with _raw suffix)
- Properties provide protocol-compliant interface
- Protocols are satisfied through structure, not inheritance
"""

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.market.enums import (
    MessageType,
    OrderBookEventType,
    OrderBookSide,
    TradeSide,
)
from src.market.model.sequences import SimplePriceLevelSequence


# Base Message Models
class CoinbaseBaseEvent(BaseModel):
    """
    Base event model for all Coinbase channel events.

    This class provides the common structure for all event types
    in the Coinbase WebSocket API.
    """

    type: str = Field(alias="type")

    model_config = ConfigDict(extra="ignore")


class CoinbaseBaseMessage(BaseModel):
    """
    Base message model for all Coinbase WebSocket messages.

    This class provides the common structure for all message types
    in the Coinbase WebSocket API.
    """

    channel: str
    client_id: str = Field(default="")
    timestamp_raw: str = Field(alias="timestamp")
    sequence_num: int
    events: list[Any]

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @property
    def exchange(self) -> str:
        """Get the exchange name."""
        return "coinbase"

    @property
    def message_type(self) -> MessageType:
        """Get the message type based on channel."""
        channel_to_type = {
            "ticker": MessageType.TICKER,
            "level2": MessageType.ORDERBOOK,
            "l2_data": MessageType.ORDERBOOK,
            "market_trades": MessageType.TRADE,
            "heartbeats": MessageType.HEARTBEAT,
            "status": MessageType.STATUS,
            "subscriptions": MessageType.SUBSCRIPTION,
            "candles": MessageType.CANDLE,
        }
        return channel_to_type.get(self.channel.lower(), MessageType.UNKNOWN)


# Ticker Channel Models
class TickerData(BaseModel):
    """
    Ticker data from Coinbase.

    Properties satisfy MarketTickerProtocol through structural typing.
    """

    type: str
    product_id: str
    price_raw: str | float = Field(alias="price")
    volume_24h_raw: str | float | None = Field(alias="volume_24h", default=None)
    low_24h_raw: str | float | None = Field(alias="low_24h", default=None)
    high_24h_raw: str | float | None = Field(alias="high_24h", default=None)
    low_52w_raw: str | float | None = Field(alias="low_52w", default=None)
    high_52w_raw: str | float | None = Field(alias="high_52w", default=None)
    price_percent_chg_24h_raw: str | float | None = Field(
        alias="price_percent_chg_24h", default=None
    )
    best_bid_raw: str | float | None = Field(alias="best_bid", default=None)
    best_ask_raw: str | float | None = Field(alias="best_ask", default=None)
    time: str

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    def _to_decimal(self, value: str | float | None) -> Decimal | None:
        """Convert string or float to Decimal, handling None."""
        if value is None:
            return None
        return Decimal(str(value))

    # Properties that satisfy MarketTickerProtocol
    @property
    def symbol(self) -> str:
        """Get market identifier."""
        return self.product_id

    @property
    def price(self) -> Decimal:
        """Get latest price."""
        return self._to_decimal(self.price_raw)  # type: ignore

    @property
    def bid(self) -> Decimal | None:
        """Get best bid price."""
        return self._to_decimal(self.best_bid_raw)

    @property
    def ask(self) -> Decimal | None:
        """Get best ask price."""
        return self._to_decimal(self.best_ask_raw)

    @property
    def volume(self) -> Decimal | None:
        """Get 24-hour trading volume."""
        return self._to_decimal(self.volume_24h_raw)

    @property
    def timestamp(self) -> datetime:
        """Get ticker timestamp."""
        return datetime.fromisoformat(self.time.replace("Z", "+00:00"))


class TickerEvent(CoinbaseBaseEvent):
    """Container for ticker updates from the ticker channel."""

    tickers: list[TickerData]

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def ensure_array_structure(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Ensure tickers is always a list."""
        if "tickers" in data:
            tickers = data["tickers"]
            if not isinstance(tickers, list):
                data["tickers"] = [tickers]
        return data


class CoinbaseTicker(CoinbaseBaseMessage):
    """
    Model for ticker channel messages.

    Contains TickerEvent with TickerData items.
    """

    events: list[TickerEvent]

    @property
    def tickers(self) -> Sequence[TickerData]:
        """
        Get ticker data items.

        Flattens all tickers from all events into a single list.
        """
        result = []
        for event in self.events:
            result.extend(event.tickers)
        return result


# Level2 (Order Book) Channel Models
class PriceLevelUpdate(BaseModel):
    """Single price level update in the order book."""

    side: str  # "bid" or "offer"
    event_time: str
    price_level_raw: str | float = Field(alias="price_level")
    new_quantity_raw: str | float = Field(alias="new_quantity")

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Helper method for consistent decimal conversion
    def _to_decimal(self, value: str | float) -> Decimal:
        """Convert string or float to Decimal."""
        return Decimal(str(value))

    @property
    def side_value(self) -> OrderBookSide:
        """Get the side as enum value."""
        return OrderBookSide.BID if self.side.lower() == "bid" else OrderBookSide.ASK

    @property
    def price_decimal(self) -> Decimal:
        """Get the price as Decimal for precision."""
        return self._to_decimal(self.price_level_raw)

    @property
    def size_decimal(self) -> Decimal:
        """Get the size as Decimal for precision."""
        return self._to_decimal(self.new_quantity_raw)


class Level2Event(CoinbaseBaseEvent):
    """Container for order book updates from level2 channel."""

    product_id: str
    updates: list[PriceLevelUpdate] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @property
    def type_value(self) -> OrderBookEventType:
        """Get the event type as enum value."""
        return (
            OrderBookEventType.SNAPSHOT
            if self.type.lower() == "snapshot"
            else OrderBookEventType.UPDATE
        )


class CoinbaseLevel2(CoinbaseBaseMessage):
    """
    Order book data from Coinbase level2 channel.

    Properties satisfy MarketOrderBookProtocol through structural typing.
    """

    events: list[Level2Event]

    # Internal cache for processed price levels
    cached_bids: list[tuple[Decimal, Decimal]] = Field(
        default_factory=list, exclude=True
    )
    cached_asks: list[tuple[Decimal, Decimal]] = Field(
        default_factory=list, exclude=True
    )

    @field_validator("channel")
    @classmethod
    def validate_level2_channel(cls, v: str) -> str:
        """Validate the channel is either level2 or l2_data."""
        if v not in ["level2", "l2_data"]:
            raise ValueError(f"Invalid channel for Level2: {v}")
        return v

    # Properties that satisfy MarketOrderBookProtocol
    @property
    def symbol(self) -> str:
        """Get market identifier."""
        if not self.events:
            raise ValueError("No level2 events available")
        return self.events[0].product_id

    @property
    def bids(self) -> SimplePriceLevelSequence:
        """
        Get bid price levels.

        If this is a snapshot, returns all bids from the snapshot.
        If this is an update, returns the updated bid price levels.
        """
        if not self.cached_bids:
            self._process_levels()
        return SimplePriceLevelSequence(self.cached_bids)

    @property
    def asks(self) -> SimplePriceLevelSequence:
        """
        Get ask price levels.

        If this is a snapshot, returns all asks from the snapshot.
        If this is an update, returns the updated ask price levels.
        """
        if not self.cached_asks:
            self._process_levels()
        return SimplePriceLevelSequence(self.cached_asks)

    @property
    def best_bid(self) -> Decimal | None:
        """
        Get best bid price.

        Returns the highest bid price or None if no bids.
        """
        if not self.cached_bids:
            self._process_levels()
        return self.cached_bids[0][0] if self.cached_bids else None

    @property
    def best_ask(self) -> Decimal | None:
        """
        Get best ask price.

        Returns the lowest ask price or None if no asks.
        """
        if not self.cached_asks:
            self._process_levels()
        return self.cached_asks[0][0] if self.cached_asks else None

    @property
    def mid_price(self) -> Decimal | None:
        """
        Get mid price between best bid and ask.

        Calculated as (best_bid + best_ask) / 2 or None if either side is empty.
        """
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> Decimal | None:
        """
        Get spread between best bid and ask.

        Calculated as best_ask - best_bid or None if either side is empty.
        """
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def spread_percentage(self) -> Decimal | None:
        """
        Get spread as percentage of mid price.

        Calculated as (spread / mid_price) * 100 or None if either side is empty.
        """
        if self.spread is None or self.mid_price is None or self.mid_price == 0:
            return None
        return (self.spread / self.mid_price) * 100

    @property
    def timestamp(self) -> datetime:
        """
        Get order book timestamp.

        Transforms Coinbase ISO-8601 timestamp to Python datetime.
        """
        return datetime.fromisoformat(self.timestamp_raw.replace("Z", "+00:00"))

    @property
    def sequence(self) -> int | None:
        """
        Get order book sequence number.

        Returns the sequence number from the message.
        """
        return self.sequence_num

    def _process_levels(self) -> None:
        """Process price levels from events."""
        # Reset the cached levels
        self.cached_bids = []
        self.cached_asks = []

        # Process all events
        for event in self.events:
            for update in event.updates:
                price = update.price_decimal
                size = update.size_decimal

                # Add to the appropriate side
                if update.side_value == OrderBookSide.BID:
                    self.cached_bids.append((price, size))
                else:
                    self.cached_asks.append((price, size))

        # Sort the levels (bids descending, asks ascending)
        self.cached_bids.sort(key=lambda x: x[0], reverse=True)
        self.cached_asks.sort(key=lambda x: x[0])


# Market Trades Channel Models
class TradeData(BaseModel):
    """
    Trade data from Coinbase.

    Properties satisfy MarketTradeProtocol through structural typing.
    """

    trade_id_raw: str | int = Field(alias="trade_id")
    product_id: str
    price_raw: str | float = Field(alias="price")
    size_raw: str | float = Field(alias="size")
    side_raw: str = Field(alias="side")  # "BUY" or "SELL"
    time: str

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    def _to_decimal(self, value: str | float) -> Decimal:
        """Convert string or float to Decimal."""
        return Decimal(str(value))

    # Properties that satisfy MarketTradeProtocol
    @property
    def symbol(self) -> str:
        """Get market identifier."""
        return self.product_id

    @property
    def price(self) -> Decimal:
        """Get trade price."""
        return self._to_decimal(self.price_raw)

    @property
    def size(self) -> Decimal:
        """Get trade size."""
        return self._to_decimal(self.size_raw)

    @property
    def side(self) -> TradeSide:
        """Get trade aggressor side."""
        return TradeSide.from_exchange(self.side_raw)

    @property
    def timestamp(self) -> datetime:
        """Get trade timestamp."""
        return datetime.fromisoformat(self.time.replace("Z", "+00:00"))

    @property
    def trade_id(self) -> str | None:
        """Get unique trade identifier."""
        return str(self.trade_id_raw) if self.trade_id_raw is not None else None


class MarketTradesEvent(CoinbaseBaseEvent):
    """Container for trade updates from market_trades channel."""

    trades: list[TradeData]

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def ensure_array_structure(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Ensure trades is always a list."""
        if "trades" in data:
            trades = data["trades"]
            if not isinstance(trades, list):
                data["trades"] = [trades]
        return data


class CoinbaseMarketTrades(CoinbaseBaseMessage):
    """
    Model for market_trades channel messages.

    Contains MarketTradesEvent with TradeData items.
    """

    events: list[MarketTradesEvent]

    @property
    def trades(self) -> Sequence[TradeData]:
        """
        Get trade data items.

        Flattens all trades from all events into a single list.
        """
        result = []
        for event in self.events:
            result.extend(event.trades)
        return result


# Heartbeats Channel Models
class HeartbeatData(BaseModel):
    """Connection health data."""

    current_time: str
    heartbeat_counter: int

    model_config = ConfigDict(extra="ignore")


class HeartbeatEvent(CoinbaseBaseEvent):
    """Heartbeat event from heartbeats channel."""

    current_time: str
    heartbeat_counter: int

    model_config = ConfigDict(extra="ignore")


class CoinbaseHeartbeat(CoinbaseBaseMessage):
    """
    Heartbeat message from Coinbase.

    Used to monitor connection health.
    """

    events: list[HeartbeatEvent]

    @property
    def current_time_datetime(self) -> datetime:
        """Get the current time as datetime."""
        if not self.events:
            return datetime.fromisoformat(self.timestamp_raw.replace("Z", "+00:00"))
        time_str = self.events[0].current_time
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))

    @property
    def heartbeat_counter_value(self) -> int:
        """Get the heartbeat counter."""
        if not self.events:
            return 0
        return self.events[0].heartbeat_counter

    # Simple properties for connection monitoring
    @property
    def sequence(self) -> int:
        """Get heartbeat sequence number."""
        return self.sequence_num

    @property
    def timestamp(self) -> datetime:
        """Get heartbeat timestamp."""
        return self.current_time_datetime


# Status Channel Models
class StatusEvent(CoinbaseBaseEvent):
    """Exchange status update from status channel."""

    status: str

    model_config = ConfigDict(extra="ignore")


class CoinbaseStatus(CoinbaseBaseMessage):
    """Status message from Coinbase."""

    events: list[StatusEvent]

    @property
    def status_value(self) -> str:
        """Get the status information."""
        if not self.events:
            return "unknown"
        return self.events[0].status


# Subscriptions Channel Models
class SubscriptionData(BaseModel):
    """Current subscription state."""

    subscriptions: dict[str, list[str]]

    model_config = ConfigDict(extra="ignore")


class CoinbaseSubscription(CoinbaseBaseMessage):
    """Subscription state message from Coinbase."""

    events: list[SubscriptionData]

    @property
    def subscriptions_map(self) -> dict[str, list[str]]:
        """Get the subscription mapping."""
        if not self.events:
            return {}
        return self.events[0].subscriptions
