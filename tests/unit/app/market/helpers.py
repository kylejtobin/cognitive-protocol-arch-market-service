"""Test helpers for market data tests."""

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.market.adapters.coinbase.data import TickerData, TradeData
from src.market.model.book import OrderBook
from src.market.model.types import PriceLevelData


def load_fixture(filename: str) -> dict[str, Any]:
    """
    Load a JSON fixture file.

    Args:
        filename: Relative path from fixtures directory

    Returns:
        Parsed JSON data

    Example:
        ticker_data = load_fixture("coinbase/ticker/btc_usd_sample.json")
    """
    fixtures_dir = Path(__file__).parent.parent.parent / "fixtures"
    fixture_path = fixtures_dir / filename

    with fixture_path.open() as f:
        data: dict[str, Any] = json.load(f)
        return data


class TickerBuilder:
    """Builder for creating test ticker data."""

    def __init__(self) -> None:
        """Initialize with sensible defaults."""
        self._data = {
            "type": "ticker",
            "product_id": "BTC-USD",
            "price": "50000.00",
            "volume_24h": "1000.0",
            "best_bid": "49999.00",
            "best_ask": "50001.00",
            "time": "2024-01-01T10:00:00Z",
        }

    def with_symbol(self, symbol: str) -> "TickerBuilder":
        """Set the product ID/symbol."""
        self._data["product_id"] = symbol
        return self

    def with_price(self, price: str | float) -> "TickerBuilder":
        """Set the price."""
        self._data["price"] = str(price)
        return self

    def with_spread(self, bid: str | float, ask: str | float) -> "TickerBuilder":
        """Set bid and ask prices."""
        self._data["best_bid"] = str(bid)
        self._data["best_ask"] = str(ask)
        return self

    def with_volume(self, volume: str | float) -> "TickerBuilder":
        """Set the 24h volume."""
        self._data["volume_24h"] = str(volume)
        return self

    def build_json(self) -> dict[str, Any]:
        """Build as raw JSON data."""
        return self._data

    def build(self) -> TickerData:
        """Build as TickerData model."""
        return TickerData.model_validate(self._data)


class OrderBookBuilder:
    """Builder for creating test order books."""

    def __init__(self) -> None:
        """Initialize with empty book."""
        self.symbol = "BTC-USD"
        self.timestamp = datetime.now(UTC)
        self.bids: list[tuple[Decimal, Decimal]] = []
        self.asks: list[tuple[Decimal, Decimal]] = []
        self.sequence: int | None = None

    def with_symbol(self, symbol: str) -> "OrderBookBuilder":
        """Set the symbol."""
        self.symbol = symbol
        return self

    def with_bid(
        self, price: float | Decimal, size: float | Decimal
    ) -> "OrderBookBuilder":
        """Add a bid level."""
        self.bids.append((Decimal(str(price)), Decimal(str(size))))
        return self

    def with_ask(
        self, price: float | Decimal, size: float | Decimal
    ) -> "OrderBookBuilder":
        """Add an ask level."""
        self.asks.append((Decimal(str(price)), Decimal(str(size))))
        return self

    def with_sequence(self, seq: int) -> "OrderBookBuilder":
        """Set the sequence number."""
        self.sequence = seq
        return self

    def build(self) -> OrderBook:
        """Build the order book."""
        # Sort bids descending, asks ascending
        bid_levels = [
            PriceLevelData(price=p, size=s) for p, s in sorted(self.bids, reverse=True)
        ]
        ask_levels = [PriceLevelData(price=p, size=s) for p, s in sorted(self.asks)]

        return OrderBook(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            sequence=self.sequence,
        )


class TradeBuilder:
    """Builder for creating test trades."""

    def __init__(self) -> None:
        """Initialize with sensible defaults."""
        self._data = {
            "trade_id": "12345",
            "product_id": "BTC-USD",
            "price": "50000.00",
            "size": "0.1",
            "side": "BUY",
            "time": "2024-01-01T10:00:00Z",
        }

    def with_symbol(self, symbol: str) -> "TradeBuilder":
        """Set the product ID/symbol."""
        self._data["product_id"] = symbol
        return self

    def with_price(self, price: str | float) -> "TradeBuilder":
        """Set the trade price."""
        self._data["price"] = str(price)
        return self

    def with_size(self, size: str | float) -> "TradeBuilder":
        """Set the trade size."""
        self._data["size"] = str(size)
        return self

    def with_side(self, side: str) -> "TradeBuilder":
        """Set the trade side (BUY/SELL)."""
        self._data["side"] = side.upper()
        return self

    def with_id(self, trade_id: str) -> "TradeBuilder":
        """Set the trade ID."""
        self._data["trade_id"] = trade_id
        return self

    def build_json(self) -> dict[str, Any]:
        """Build as raw JSON data."""
        return self._data

    def build(self) -> TradeData:
        """Build as TradeData model."""
        return TradeData.model_validate(self._data)


def create_ticker_message(
    symbol: str = "BTC-USD", **ticker_fields: Any
) -> dict[str, Any]:
    """
    Create a complete ticker WebSocket message.

    Args:
        symbol: The product ID
        **ticker_fields: Additional fields to override in the ticker

    Returns:
        Complete WebSocket message ready to emit
    """
    ticker_data = TickerBuilder().with_symbol(symbol).build_json()
    ticker_data.update(ticker_fields)

    return {
        "channel": "ticker",
        "client_id": "",
        "timestamp": "2024-01-01T10:00:00Z",
        "sequence_num": 1,
        "events": [{"type": "ticker", "tickers": [ticker_data]}],
    }


def create_level2_snapshot(
    symbol: str = "BTC-USD",
    bids: list[tuple[float, float]] | None = None,
    asks: list[tuple[float, float]] | None = None,
    sequence: int = 100,
) -> dict[str, Any]:
    """
    Create a level2 order book snapshot message.

    Args:
        symbol: The product ID
        bids: List of (price, size) tuples for bids
        asks: List of (price, size) tuples for asks
        sequence: Sequence number

    Returns:
        Complete WebSocket message ready to emit
    """
    if bids is None:
        bids = [(50000.0, 1.0), (49999.0, 2.0)]
    if asks is None:
        asks = [(50001.0, 1.0), (50002.0, 2.0)]

    updates = []
    for price, size in bids:
        updates.append(
            {
                "side": "bid",
                "event_time": "2024-01-01T10:00:00Z",
                "price_level": str(price),
                "new_quantity": str(size),
            }
        )

    for price, size in asks:
        updates.append(
            {
                "side": "offer",
                "event_time": "2024-01-01T10:00:00Z",
                "price_level": str(price),
                "new_quantity": str(size),
            }
        )

    return {
        "channel": "level2",
        "client_id": "",
        "timestamp": "2024-01-01T10:00:00Z",
        "sequence_num": sequence,
        "events": [{"type": "snapshot", "product_id": symbol, "updates": updates}],
    }
