"""Test WebSocket message flow with a mock client."""

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pytest

from src.market.adapters.coinbase.stream import CoinbaseStreamHandler
from src.market.model.snapshot import MarketSnapshot


class MockWSClient:
    """
    Mock WebSocket client for testing without network calls.

    This allows us to:
    - Simulate incoming messages
    - Test message parsing and state updates
    - Verify callback behavior
    - Test error scenarios
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        on_message: Callable[[str], None] | None = None,
        retry: bool = True,
        verbose: bool = False,
    ):
        """Initialize mock client matching Coinbase WSClient interface."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.on_message = on_message
        self.retry = retry
        self.verbose = verbose

        # Track state
        self.is_open = False
        self.subscriptions: dict[str, list[str]] = {}

        # Mock methods
        self.open = Mock(side_effect=self._open)
        self.close = Mock(side_effect=self._close)
        self.ticker = Mock(side_effect=self._subscribe_ticker)
        self.level2 = Mock(side_effect=self._subscribe_level2)
        self.market_trades = Mock(side_effect=self._subscribe_trades)

    def _open(self) -> None:
        """Simulate opening connection."""
        self.is_open = True

    def _close(self) -> None:
        """Simulate closing connection."""
        self.is_open = False

    def _subscribe_ticker(self, product_ids: list[str]) -> None:
        """Track ticker subscriptions."""
        self.subscriptions["ticker"] = product_ids

    def _subscribe_level2(self, product_ids: list[str]) -> None:
        """Track level2 subscriptions."""
        self.subscriptions["level2"] = product_ids

    def _subscribe_trades(self, product_ids: list[str]) -> None:
        """Track market_trades subscriptions."""
        self.subscriptions["market_trades"] = product_ids

    def emit_message(self, message: dict[str, Any]) -> None:
        """
        Simulate receiving a message from the WebSocket.

        This is the key method for testing - it allows us to
        simulate any message we want and verify the handler processes it.
        """
        if self.on_message and self.is_open:
            self.on_message(json.dumps(message))

    def sleep_with_exception_check(self, sleep: int) -> None:
        """Mock sleep method."""
        pass


class TestWebSocketMessageFlow:
    """Test the complete message flow from WebSocket to MarketSnapshot."""

    @pytest.fixture
    def mock_client_class(self) -> type[MockWSClient]:
        """Return the mock WebSocket client class."""
        return MockWSClient

    @pytest.fixture
    def snapshot_callback(self) -> Mock:
        """Create a mock callback to track snapshots."""
        return Mock(spec=Callable[[MarketSnapshot], None])

    def test_ticker_message_creates_snapshot(
        self,
        mock_client_class: type[MockWSClient],
        snapshot_callback: Mock,
    ) -> None:
        """Test that ticker messages result in market snapshots."""
        # Given: A stream handler with our callback
        handler = CoinbaseStreamHandler(on_snapshot=snapshot_callback)

        # And: A mock WebSocket client
        client = mock_client_class(on_message=handler.handle_message)
        client.open()

        # When: We receive a ticker message
        ticker_message = {
            "channel": "ticker",
            "client_id": "",
            "timestamp": "2024-01-01T10:00:00Z",
            "sequence_num": 1,
            "events": [
                {
                    "type": "ticker",
                    "tickers": [
                        {
                            "type": "ticker",
                            "product_id": "BTC-USD",
                            "price": "45234.50",
                            "volume_24h": "1234.5678",
                            "best_bid": "45230.00",
                            "best_ask": "45235.00",
                            "time": "2024-01-01T10:00:00Z",
                        }
                    ],
                }
            ],
        }
        client.emit_message(ticker_message)

        # Then: A snapshot should be created
        snapshot_callback.assert_called_once()

        # And: The snapshot should contain the ticker data
        snapshot = snapshot_callback.call_args[0][0]
        assert isinstance(snapshot, MarketSnapshot)
        assert snapshot.symbol == "BTC-USD"
        assert snapshot.ticker is not None
        assert str(snapshot.ticker.price) == "45234.50"

    def test_order_book_snapshot_and_update(
        self,
        mock_client_class: type[MockWSClient],
        snapshot_callback: Mock,
    ) -> None:
        """Test order book snapshot followed by incremental update."""
        # Given: A stream handler
        handler = CoinbaseStreamHandler(on_snapshot=snapshot_callback)
        client = mock_client_class(on_message=handler.handle_message)
        client.open()

        # When: We receive an order book snapshot
        book_snapshot = {
            "channel": "level2",
            "client_id": "",
            "timestamp": "2024-01-01T10:00:00Z",
            "sequence_num": 100,
            "events": [
                {
                    "type": "snapshot",
                    "product_id": "ETH-USD",
                    "updates": [
                        {
                            "side": "bid",
                            "event_time": "2024-01-01T10:00:00Z",
                            "price_level": "2000.00",
                            "new_quantity": "10.0",
                        },
                        {
                            "side": "offer",
                            "event_time": "2024-01-01T10:00:00Z",
                            "price_level": "2001.00",
                            "new_quantity": "10.0",
                        },
                    ],
                }
            ],
        }
        client.emit_message(book_snapshot)

        # Then: First snapshot created
        assert snapshot_callback.call_count == 1
        first_snapshot = snapshot_callback.call_args_list[0][0][0]
        assert first_snapshot.order_book is not None
        assert str(first_snapshot.order_book.best_bid) == "2000.00"
        assert str(first_snapshot.order_book.best_ask) == "2001.00"

        # When: We receive an update
        book_update = {
            "channel": "level2",
            "client_id": "",
            "timestamp": "2024-01-01T10:00:01Z",
            "sequence_num": 101,
            "events": [
                {
                    "type": "update",
                    "product_id": "ETH-USD",
                    "updates": [
                        {
                            "side": "bid",
                            "event_time": "2024-01-01T10:00:01Z",
                            "price_level": "2000.50",
                            "new_quantity": "5.0",
                        },
                    ],
                }
            ],
        }
        client.emit_message(book_update)

        # Then: Second snapshot created with updated book
        assert snapshot_callback.call_count == 2
        second_snapshot = snapshot_callback.call_args_list[1][0][0]
        assert second_snapshot.order_book is not None
        # Original best bid should still exist
        assert str(second_snapshot.order_book.best_bid) == "2000.50"

    def test_multiple_symbols_tracked_independently(
        self,
        mock_client_class: type[MockWSClient],
        snapshot_callback: Mock,
    ) -> None:
        """Test that multiple symbols maintain independent state."""
        # Given: A stream handler
        handler = CoinbaseStreamHandler(on_snapshot=snapshot_callback)
        client = mock_client_class(on_message=handler.handle_message)
        client.open()

        # When: We receive tickers for different symbols
        for symbol in ["BTC-USD", "ETH-USD", "SOL-USD"]:
            ticker_message = {
                "channel": "ticker",
                "client_id": "",
                "timestamp": "2024-01-01T10:00:00Z",
                "sequence_num": 1,
                "events": [
                    {
                        "type": "ticker",
                        "tickers": [
                            {
                                "type": "ticker",
                                "product_id": symbol,
                                "price": "100.00",
                                "time": "2024-01-01T10:00:00Z",
                            }
                        ],
                    }
                ],
            }
            client.emit_message(ticker_message)

        # Then: Three separate snapshots created
        assert snapshot_callback.call_count == 3

        # And: Each has the correct symbol
        symbols = [call[0][0].symbol for call in snapshot_callback.call_args_list]
        assert sorted(symbols) == ["BTC-USD", "ETH-USD", "SOL-USD"]

    def test_malformed_message_handled_gracefully(
        self,
        mock_client_class: type[MockWSClient],
        snapshot_callback: Mock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that malformed messages don't crash the handler."""
        # Given: A stream handler
        handler = CoinbaseStreamHandler(on_snapshot=snapshot_callback)
        client = mock_client_class(on_message=handler.handle_message)
        client.open()

        # When: We send various malformed messages
        malformed_messages = [
            {},  # Empty message
            {"channel": "unknown_channel"},  # Unknown channel
            {"channel": "ticker"},  # Missing required fields
            {"channel": "ticker", "events": "not_a_list"},  # Wrong type
        ]

        for msg in malformed_messages:
            client.emit_message(msg)

        # Then: No snapshots created
        snapshot_callback.assert_not_called()

        # And: Errors were printed (not exceptions raised)
        captured = capsys.readouterr()
        assert "Error" in captured.out or len(captured.out) > 0

    def test_subscription_tracking(
        self,
        mock_client_class: type[MockWSClient],
    ) -> None:
        """Test that subscriptions are properly tracked."""
        # Given: A mock client
        client = mock_client_class()

        # When: We subscribe to channels
        symbols = ["BTC-USD", "ETH-USD"]
        client.open()
        client.ticker(product_ids=symbols)
        client.level2(product_ids=symbols)
        client.market_trades(product_ids=symbols)

        # Then: Subscriptions are tracked
        assert client.subscriptions["ticker"] == symbols
        assert client.subscriptions["level2"] == symbols
        assert client.subscriptions["market_trades"] == symbols

        # And: Methods were called
        client.ticker.assert_called_once_with(product_ids=symbols)
        client.level2.assert_called_once_with(product_ids=symbols)
        client.market_trades.assert_called_once_with(product_ids=symbols)
