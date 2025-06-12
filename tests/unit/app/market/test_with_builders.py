"""Tests using builder pattern for clarity and maintainability."""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.market.adapters.coinbase.stream import CoinbaseStreamHandler
from src.market.model.snapshot import MarketSnapshot
from tests.unit.app.market.helpers import (
    OrderBookBuilder,
    TickerBuilder,
    TradeBuilder,
    create_level2_snapshot,
    create_ticker_message,
)
from tests.unit.app.market.test_mock_websocket import MockWSClient


class TestMarketDataWithBuilders:
    """Demonstrate testing with builders for better readability."""

    def test_ticker_builder_creates_valid_data(self) -> None:
        """Test that our builder creates valid ticker data."""
        # Given: A ticker built with specific values
        ticker = (
            TickerBuilder()
            .with_symbol("ETH-USD")
            .with_price(2500.50)
            .with_spread(2499.00, 2502.00)
            .with_volume(5000)
            .build()
        )

        # Then: All values are correctly set and converted
        assert ticker.symbol == "ETH-USD"
        assert ticker.price == Decimal("2500.50")
        assert ticker.bid == Decimal("2499.00")
        assert ticker.ask == Decimal("2502.00")
        assert ticker.volume == Decimal("5000")

    def test_order_book_builder_sorts_levels(self) -> None:
        """Test that order book builder properly sorts price levels."""
        # Given: An order book with unsorted levels
        book = (
            OrderBookBuilder()
            .with_symbol("BTC-USD")
            .with_bid(45000, 1.0)
            .with_bid(45100, 2.0)  # Higher bid added second
            .with_bid(44900, 3.0)  # Lower bid added third
            .with_ask(45300, 1.0)
            .with_ask(45200, 2.0)  # Lower ask added second
            .with_ask(45400, 3.0)  # Higher ask added third
            .build()
        )

        # Then: Bids are sorted descending (best first)
        assert book.bids.get_price_at_index(0) == Decimal("45100")
        assert book.bids.get_price_at_index(1) == Decimal("45000")
        assert book.bids.get_price_at_index(2) == Decimal("44900")

        # And: Asks are sorted ascending (best first)
        assert book.asks.get_price_at_index(0) == Decimal("45200")
        assert book.asks.get_price_at_index(1) == Decimal("45300")
        assert book.asks.get_price_at_index(2) == Decimal("45400")

    def test_complex_market_scenario(self) -> None:
        """Test a complex scenario with multiple data types."""
        # Given: A mock WebSocket and handler
        snapshot_callback = Mock()
        handler = CoinbaseStreamHandler(on_snapshot=snapshot_callback)
        client = MockWSClient(on_message=handler.handle_message)
        client.open()

        # When: We simulate a realistic sequence of messages

        # 1. Initial order book snapshot
        client.emit_message(
            create_level2_snapshot(
                symbol="BTC-USD",
                bids=[(45000, 10), (44999, 20), (44998, 30)],
                asks=[(45001, 10), (45002, 20), (45003, 30)],
                sequence=1000,
            )
        )

        # 2. Ticker update with mid-market price
        client.emit_message(
            create_ticker_message(
                symbol="BTC-USD",
                price="45000.50",
                best_bid="45000.00",
                best_ask="45001.00",
            )
        )

        # Then: We should have two snapshots
        assert snapshot_callback.call_count == 2

        # And: The second snapshot should have both ticker and order book
        final_snapshot: MarketSnapshot = snapshot_callback.call_args_list[1][0][0]
        assert final_snapshot.symbol == "BTC-USD"
        assert final_snapshot.ticker is not None
        assert final_snapshot.order_book is not None

        # And: Values should match what we sent
        assert final_snapshot.ticker.price == Decimal("45000.50")
        assert final_snapshot.order_book.best_bid == Decimal("45000")
        assert final_snapshot.order_book.best_ask == Decimal("45001")
        assert final_snapshot.order_book.spread == Decimal("1")

    def test_trade_aggregation_scenario(self) -> None:
        """Test trade aggregation with builders."""
        # Given: Multiple trades at different prices
        trades = [
            TradeBuilder()
            .with_symbol("SOL-USD")
            .with_price(100 + i)
            .with_size(0.1 * i)
            .with_side("BUY" if i % 2 == 0 else "SELL")
            .with_id(f"trade-{i}")
            .build()
            for i in range(1, 6)
        ]

        # Then: We can analyze the trades
        buy_trades = [t for t in trades if t.side.value == "buy"]
        sell_trades = [t for t in trades if t.side.value == "sell"]

        assert len(buy_trades) == 2  # Trades 2 and 4
        assert len(sell_trades) == 3  # Trades 1, 3, and 5

        # And: Calculate volume-weighted average price (VWAP)
        total_value = sum(t.price * t.size for t in trades)
        total_volume = sum(t.size for t in trades)
        vwap = total_value / total_volume

        # The VWAP should be reasonable
        assert Decimal("102") < vwap < Decimal("104")

    @pytest.mark.parametrize(
        "price,expected_spread_pct",
        [
            (1000, Decimal("0.2")),  # $2 spread on $1000 = 0.2%
            (10000, Decimal("0.02")),  # $2 spread on $10000 = 0.02%
            (100, Decimal("2.0")),  # $2 spread on $100 = 2.0%
        ],
    )
    def test_spread_percentage_calculations(
        self,
        price: float,
        expected_spread_pct: Decimal,
    ) -> None:
        """Test spread percentage calculation at different price levels."""
        # Given: An order book with $2 spread at different price levels
        book = (
            OrderBookBuilder().with_bid(price - 1, 1.0).with_ask(price + 1, 1.0).build()
        )

        # Then: Spread percentage should be calculated correctly
        assert book.spread == Decimal("2")
        assert book.spread_percentage == pytest.approx(expected_spread_pct, rel=1e-6)
