"""
Event models for market data streaming.

These models provide type-safe representations of events that flow
through the system, enabling proper event-driven architecture.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from src.market.protocols import (
    MarketOrderBookProtocol,
    MarketTickerProtocol,
    MarketTradeProtocol,
)


class MarketDataEvent(BaseModel):
    """Base event for all market data updates."""

    event_type: Literal[
        "snapshot", "ticker", "trade", "orderbook", "error", "connected", "disconnected"
    ]
    symbol: str
    timestamp: datetime
    sequence: int | None = Field(
        default=None, description="Optional sequence number for ordering"
    )

    def to_log_entry(self) -> str:
        """Generate a log-friendly representation."""
        return (
            f"[{self.event_type.upper()}] {self.symbol} @ {self.timestamp.isoformat()}"
        )


class TickerUpdateEvent(MarketDataEvent):
    """Ticker price update event."""

    event_type: Literal["ticker"] = "ticker"
    ticker: MarketTickerProtocol = Field(description="The ticker data")

    @property
    def price_change(self) -> float | None:
        """Calculate price change if previous price available."""
        # This would need previous price tracking in practice
        return None

    def is_significant_move(self, threshold_percent: float = 1.0) -> bool:
        """Check if this represents a significant price move."""
        # Simplified - would need historical context
        return False


class OrderBookUpdateEvent(MarketDataEvent):
    """Order book state update event."""

    event_type: Literal["orderbook"] = "orderbook"
    order_book: MarketOrderBookProtocol = Field(description="The order book data")
    update_type: Literal["snapshot", "delta"] = Field(
        default="snapshot",
        description="Whether this is a full snapshot or incremental update",
    )

    @property
    def spread_bps(self) -> float | None:
        """Get spread in basis points."""
        if not self.order_book.spread_percentage:
            return None
        return float(self.order_book.spread_percentage * 100)

    def has_liquidity(self, min_depth: int = 5) -> bool:
        """Check if order book has minimum liquidity."""
        return (
            len(self.order_book.bids) >= min_depth
            and len(self.order_book.asks) >= min_depth
        )


class TradeExecutedEvent(MarketDataEvent):
    """Trade execution event."""

    event_type: Literal["trade"] = "trade"
    trade: MarketTradeProtocol = Field(description="The trade data")

    @property
    def is_large_trade(self) -> bool:
        """Determine if this is a large trade."""
        # Would need symbol-specific thresholds
        return float(self.trade.size) > 1.0  # Simplified

    def to_summary(self) -> str:
        """Generate trade summary."""
        return (
            f"{self.trade.side.value} {self.trade.size} @ "
            f"{self.trade.price} on {self.symbol}"
        )


class MarketSnapshotEvent(MarketDataEvent):
    """Complete market state snapshot event."""

    event_type: Literal["snapshot"] = "snapshot"
    ticker: MarketTickerProtocol | None = None
    order_book: MarketOrderBookProtocol | None = None
    recent_trades: list[MarketTradeProtocol] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Check if snapshot has all components."""
        return self.ticker is not None and self.order_book is not None

    def to_analysis_input(self) -> dict[str, float | None]:
        """Convert to format suitable for analysis."""
        return {
            "price": float(self.ticker.price) if self.ticker else None,
            "bid": float(self.order_book.best_bid)
            if self.order_book and self.order_book.best_bid
            else None,
            "ask": float(self.order_book.best_ask)
            if self.order_book and self.order_book.best_ask
            else None,
            "spread": float(self.order_book.spread)
            if self.order_book and self.order_book.spread
            else None,
            "volume": len(self.recent_trades),
        }


class ConnectionEvent(MarketDataEvent):
    """WebSocket connection state event."""

    event_type: Literal["connected", "disconnected"] = "connected"
    symbol: str = Field(default="SYSTEM", description="System-level event")
    reason: str | None = Field(
        default=None, description="Disconnection reason if applicable"
    )
    reconnect_in: int | None = Field(
        default=None, description="Seconds until reconnect attempt"
    )

    def should_reconnect(self) -> bool:
        """Determine if automatic reconnection should be attempted."""
        return self.event_type == "disconnected" and self.reconnect_in is not None


class ErrorEvent(MarketDataEvent):
    """Error event for exceptional conditions."""

    event_type: Literal["error"] = "error"
    error_code: str = Field(description="Error code for categorization")
    error_message: str = Field(description="Human-readable error message")
    recoverable: bool = Field(default=True, description="Whether error is recoverable")
    context: dict[str, str] = Field(
        default_factory=dict, description="Additional error context"
    )

    def to_alert(self) -> str:
        """Format as alert message."""
        severity = "WARNING" if self.recoverable else "ERROR"
        return f"[{severity}] {self.error_code}: {self.error_message}"


class EventBatch(BaseModel):
    """Batch of events for efficient processing."""

    events: list[MarketDataEvent] = Field(description="Ordered list of events")
    batch_id: str = Field(description="Unique batch identifier")
    created_at: datetime = Field(description="Batch creation timestamp")

    @property
    def size(self) -> int:
        """Get number of events in batch."""
        return len(self.events)

    @property
    def symbols(self) -> set[str]:
        """Get unique symbols in batch."""
        return {event.symbol for event in self.events}

    def filter_by_type(self, event_type: str) -> list[MarketDataEvent]:
        """Filter events by type."""
        return [e for e in self.events if e.event_type == event_type]

    def to_metrics(self) -> dict[str, int]:
        """Generate batch metrics."""
        metrics: dict[str, int] = {}
        for event in self.events:
            key = f"events_{event.event_type}"
            metrics[key] = metrics.get(key, 0) + 1
        return metrics
