"""
Coinbase WebSocket streaming integration.

This module integrates with the Coinbase Advanced Trade WebSocket client,
handling message parsing and state management for market data streaming.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from coinbase.websocket import WSClient

from src.market.adapters.coinbase.book_manager import CoinbaseBookManager
from src.market.adapters.coinbase.data import (
    CoinbaseLevel2,
    CoinbaseMarketTrades,
    CoinbaseTicker,
)
from src.market.model.snapshot import MarketSnapshot
from src.market.model.ticker import MarketTicker
from src.market.model.trade import MarketTrade

if TYPE_CHECKING:
    pass


class CoinbaseStreamHandler:
    """
    Handles Coinbase WebSocket messages and maintains market state.

    This class:
    - Parses incoming WebSocket messages using Pydantic models
    - Maintains order book state with the book manager
    - Creates MarketSnapshot instances on updates
    - Handles errors gracefully
    """

    def __init__(self, on_snapshot: Callable[[MarketSnapshot], None]) -> None:
        """
        Initialize the stream handler.

        Args:
            on_snapshot: Callback for when a market snapshot is updated

        """
        self.on_snapshot = on_snapshot
        self.book_manager = CoinbaseBookManager()
        self.latest_tickers: dict[str, MarketTicker] = {}
        self.recent_trades: dict[str, list[MarketTrade]] = {}
        self.max_trades = 100

    def handle_message(self, msg: str) -> None:
        """
        Handle raw WebSocket message from Coinbase.

        This is the main entry point called by the WebSocket client.
        """
        try:
            # Parse JSON to discover channel type
            data = json.loads(msg)
            channel = data.get("channel", "")

            # Route to appropriate typed handler based on channel
            match channel:
                case "ticker" | "ticker_batch":
                    self._handle_ticker_message(msg)
                case "level2" | "l2_data":
                    self._handle_level2_message(msg)
                case "market_trades":
                    self._handle_trades_message(msg)
                case "subscriptions" | "heartbeats":
                    # Ignore these channels for now
                    pass
                case _:
                    if data.get("type") != "error":
                        print(f"Unknown channel: {channel}")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON message: {e}")
        except Exception as e:
            print(f"Error routing message: {e}")

    def _handle_ticker_message(self, msg: str) -> None:
        """Handle ticker messages with typed parsing."""
        try:
            # Parse directly into typed model
            ticker_msg = CoinbaseTicker.model_validate_json(msg)

            for event in ticker_msg.events:
                for ticker_data in event.tickers:
                    symbol = ticker_data.product_id

                    # Transform to domain model
                    domain_ticker = MarketTicker(
                        symbol=symbol,
                        exchange="coinbase",
                        price=ticker_data.price,
                        size=Decimal("0"),  # Ticker doesn't include last trade size
                        bid=ticker_data.bid,
                        ask=ticker_data.ask,
                        bid_size=None,  # Not provided by ticker channel
                        ask_size=None,  # Not provided by ticker channel
                        volume=ticker_data.volume,
                        vwap=None,  # Not provided by ticker channel
                        timestamp=ticker_data.timestamp,
                    )

                    self.latest_tickers[symbol] = domain_ticker
                    self._emit_snapshot(symbol)
        except Exception as e:
            print(f"Error handling ticker: {e}")

    def _handle_level2_message(self, msg: str) -> None:
        """Handle level2 order book messages with typed parsing."""
        try:
            # Parse directly into typed model
            level2_msg = CoinbaseLevel2.model_validate_json(msg)
            updated_books = self.book_manager.process_level2_message(level2_msg)

            for book in updated_books:
                self._emit_snapshot(book.symbol)
        except Exception as e:
            print(f"Error handling level2: {e}")

    def _handle_trades_message(self, msg: str) -> None:
        """Handle market trades messages with typed parsing."""
        try:
            # Parse directly into typed model
            trades_msg = CoinbaseMarketTrades.model_validate_json(msg)

            for event in trades_msg.events:
                for trade_data in event.trades:
                    symbol = trade_data.product_id

                    # Transform to domain model
                    domain_trade = MarketTrade(
                        symbol=symbol,
                        price=trade_data.price,
                        size=trade_data.size,
                        side=trade_data.side,
                        timestamp=trade_data.timestamp,
                        trade_id=trade_data.trade_id,
                    )

                    # Initialize trades list if needed
                    if symbol not in self.recent_trades:
                        self.recent_trades[symbol] = []

                    # Add new trade and maintain max size
                    self.recent_trades[symbol].insert(0, domain_trade)
                    if len(self.recent_trades[symbol]) > self.max_trades:
                        self.recent_trades[symbol] = self.recent_trades[symbol][
                            : self.max_trades
                        ]

                    self._emit_snapshot(symbol)
        except Exception as e:
            print(f"Error handling trades: {e}")

    def _emit_snapshot(self, symbol: str) -> None:
        """Create and emit a market snapshot for a symbol."""
        # Get current order book
        book = self.book_manager.get_book(symbol)
        order_book = book.to_protocol() if book else None

        # Get current ticker
        ticker = self.latest_tickers.get(symbol)

        # Get recent trades
        trades = self.recent_trades.get(symbol, [])

        # Create snapshot
        snapshot = MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            ticker=ticker,
            order_book=order_book,
            trades=trades,
        )

        # Emit via callback
        self.on_snapshot(snapshot)


def stream_coinbase_market_data(
    symbols: list[str],
    on_snapshot: Callable[[MarketSnapshot], None],
    api_key: str | None = None,
    api_secret: str | None = None,
) -> WSClient:
    """
    Stream market data from Coinbase WebSocket.

    Args:
        symbols: List of product IDs to stream (e.g., ["BTC-USD", "ETH-USD"])
        on_snapshot: Callback for market snapshot updates
        api_key: Optional Coinbase API key
        api_secret: Optional Coinbase API secret

    Returns:
        WSClient instance for connection management

    """
    # Create handler
    handler = CoinbaseStreamHandler(on_snapshot)

    # Create WebSocket client
    client = WSClient(
        api_key=api_key,
        api_secret=api_secret,
        on_message=handler.handle_message,
        retry=True,  # Auto-reconnect
        verbose=False,  # Set to True for debugging
    )

    # Open connection
    client.open()

    # Subscribe to channels
    # Note: Coinbase client methods are synchronous despite async naming
    client.ticker(product_ids=symbols)
    client.level2(product_ids=symbols)
    client.market_trades(product_ids=symbols)

    return client
