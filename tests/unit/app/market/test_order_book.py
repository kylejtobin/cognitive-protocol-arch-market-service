"""Test order book management and price level sequences."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.market.adapters.coinbase.book_manager import CoinbaseBookManager
from src.market.adapters.coinbase.data import CoinbaseLevel2
from src.market.model.book import MutableOrderBook, OrderBook, PriceLevelSequence
from src.market.model.sequences import SimplePriceLevelSequence
from src.market.model.types import PriceLevelData


class TestPriceLevelSequence:
    """Test price level sequence functionality."""

    def test_price_level_sequence_sorting(self) -> None:
        """Test that price levels are sorted correctly."""
        # Given: Unsorted price levels
        levels = [
            PriceLevelData(price=Decimal("100.50"), size=Decimal("1.0")),
            PriceLevelData(price=Decimal("100.00"), size=Decimal("2.0")),
            PriceLevelData(price=Decimal("101.00"), size=Decimal("3.0")),
        ]

        # When: Creating sequences for bids and asks
        bid_sequence = PriceLevelSequence(levels, reverse=True)  # Descending
        ask_sequence = PriceLevelSequence(levels, reverse=False)  # Ascending

        # Then: Bids should be sorted descending
        assert bid_sequence.get_price_at_index(0) == Decimal("101.00")
        assert bid_sequence.get_price_at_index(1) == Decimal("100.50")
        assert bid_sequence.get_price_at_index(2) == Decimal("100.00")

        # And: Asks should be sorted ascending
        assert ask_sequence.get_price_at_index(0) == Decimal("100.00")
        assert ask_sequence.get_price_at_index(1) == Decimal("100.50")
        assert ask_sequence.get_price_at_index(2) == Decimal("101.00")

    def test_simple_price_level_sequence(self) -> None:
        """Test SimplePriceLevelSequence functionality."""
        # Given: Price levels as tuples
        levels = [
            (Decimal("100.00"), Decimal("1.0")),
            (Decimal("99.00"), Decimal("2.0")),
        ]

        # When: Creating a simple sequence
        sequence = SimplePriceLevelSequence(levels)

        # Then: Properties should work correctly
        assert len(sequence) == 2
        assert sequence.best_price == Decimal("100.00")
        assert sequence.get_total_size() == Decimal("3.0")

        # And: Individual level access should work
        level = sequence[0]
        assert level.price == Decimal("100.00")
        assert level.size == Decimal("1.0")


class TestOrderBook:
    """Test OrderBook model functionality."""

    def test_order_book_properties(self) -> None:
        """Test that OrderBook properties calculate correctly."""
        # Given: Bid and ask levels
        bid_levels = [
            PriceLevelData(price=Decimal("100.00"), size=Decimal("1.0")),
            PriceLevelData(price=Decimal("99.50"), size=Decimal("2.0")),
        ]
        ask_levels = [
            PriceLevelData(price=Decimal("100.50"), size=Decimal("1.0")),
            PriceLevelData(price=Decimal("101.00"), size=Decimal("2.0")),
        ]

        # When: Creating an order book
        book = OrderBook(
            symbol="TEST-USD",
            timestamp=datetime.now(UTC),
            bid_levels=bid_levels,
            ask_levels=ask_levels,
        )

        # Then: Properties should calculate correctly
        assert book.best_bid == Decimal("100.00")
        assert book.best_ask == Decimal("100.50")
        assert book.mid_price == Decimal("100.25")
        assert book.spread == Decimal("0.50")
        assert float(book.spread_percentage) == pytest.approx(0.4988, rel=1e-4)

        # And: Price level sequences should work
        assert len(book.bids) == 2
        assert len(book.asks) == 2
        assert book.bids.best_price == Decimal("100.00")
        assert book.asks.best_price == Decimal("100.50")


class TestMutableOrderBook:
    """Test MutableOrderBook for incremental updates."""

    def test_apply_snapshot(self) -> None:
        """Test applying a snapshot to a mutable order book."""
        # Given: A mutable order book
        book = MutableOrderBook("TEST-USD")

        # And: Snapshot data
        bids = [
            PriceLevelData(price=Decimal("100.00"), size=Decimal("1.0")),
            PriceLevelData(price=Decimal("99.00"), size=Decimal("2.0")),
        ]
        asks = [
            PriceLevelData(price=Decimal("101.00"), size=Decimal("1.0")),
            PriceLevelData(price=Decimal("102.00"), size=Decimal("2.0")),
        ]

        # When: Applying the snapshot
        book.apply_snapshot(bids, asks, sequence=123)

        # Then: Book should contain the snapshot data
        assert book.bids[Decimal("100.00")] == Decimal("1.0")
        assert book.bids[Decimal("99.00")] == Decimal("2.0")
        assert book.asks[Decimal("101.00")] == Decimal("1.0")
        assert book.asks[Decimal("102.00")] == Decimal("2.0")
        assert book.sequence == 123

    def test_apply_update(self) -> None:
        """Test applying incremental updates."""
        # Given: A book with initial state
        book = MutableOrderBook("TEST-USD")
        book.bids[Decimal("100.00")] = Decimal("1.0")
        book.asks[Decimal("101.00")] = Decimal("1.0")

        # When: Updating a price level
        book.apply_update("bid", Decimal("100.00"), Decimal("2.0"))

        # Then: Size should be updated
        assert book.bids[Decimal("100.00")] == Decimal("2.0")

        # When: Removing a price level (size = 0)
        book.apply_update("ask", Decimal("101.00"), Decimal("0"))

        # Then: Level should be removed
        assert Decimal("101.00") not in book.asks

    def test_to_protocol(self) -> None:
        """Test converting mutable book to immutable protocol version."""
        # Given: A mutable book with data
        mutable = MutableOrderBook("TEST-USD")
        mutable.bids[Decimal("100.00")] = Decimal("1.0")
        mutable.bids[Decimal("99.00")] = Decimal("2.0")
        mutable.asks[Decimal("101.00")] = Decimal("1.0")
        mutable.sequence = 456

        # When: Converting to protocol
        immutable = mutable.to_protocol()

        # Then: Should create proper OrderBook
        assert immutable.symbol == "TEST-USD"
        assert immutable.best_bid == Decimal("100.00")
        assert immutable.best_ask == Decimal("101.00")
        assert immutable.sequence == 456
        assert len(immutable.bid_levels) == 2
        assert len(immutable.ask_levels) == 1


class TestCoinbaseBookManager:
    """Test Coinbase-specific book management."""

    def test_process_snapshot(self) -> None:
        """Test processing a level2 snapshot message."""
        # Given: A book manager
        manager = CoinbaseBookManager()

        # And: A snapshot message
        message_data = {
            "channel": "level2",
            "client_id": "",
            "timestamp": "2024-01-01T10:00:00Z",
            "sequence_num": 1,
            "events": [
                {
                    "type": "snapshot",
                    "product_id": "BTC-USD",
                    "updates": [
                        {
                            "side": "bid",
                            "event_time": "2024-01-01T10:00:00Z",
                            "price_level": "45230.00",
                            "new_quantity": "1.0",
                        },
                        {
                            "side": "offer",
                            "event_time": "2024-01-01T10:00:00Z",
                            "price_level": "45235.00",
                            "new_quantity": "1.0",
                        },
                    ],
                }
            ],
        }
        message = CoinbaseLevel2.model_validate(message_data)

        # When: Processing the message
        updated_books = manager.process_level2_message(message)

        # Then: Should create a book for BTC-USD
        assert len(updated_books) == 1
        book = updated_books[0]
        assert book.symbol == "BTC-USD"
        assert book.bids[Decimal("45230.00")] == Decimal("1.0")
        assert book.asks[Decimal("45235.00")] == Decimal("1.0")

        # And: Manager should track the book
        assert "BTC-USD" in manager.books
        assert manager.expected_sequences["BTC-USD"] == 2

    def test_process_update(self) -> None:
        """Test processing incremental updates."""
        # Given: A manager with an existing book
        manager = CoinbaseBookManager()
        manager.books["ETH-USD"] = MutableOrderBook("ETH-USD")
        manager.books["ETH-USD"].bids[Decimal("2000.00")] = Decimal("1.0")
        manager.expected_sequences["ETH-USD"] = 10

        # And: An update message
        message_data = {
            "channel": "level2",
            "client_id": "",
            "timestamp": "2024-01-01T10:00:00Z",
            "sequence_num": 10,
            "events": [
                {
                    "type": "update",
                    "product_id": "ETH-USD",
                    "updates": [
                        {
                            "side": "bid",
                            "event_time": "2024-01-01T10:00:00Z",
                            "price_level": "2000.00",
                            "new_quantity": "2.0",
                        },
                    ],
                }
            ],
        }
        message = CoinbaseLevel2.model_validate(message_data)

        # When: Processing the update
        updated_books = manager.process_level2_message(message)

        # Then: Should update the existing book
        assert len(updated_books) == 1
        book = updated_books[0]
        assert book.bids[Decimal("2000.00")] == Decimal("2.0")
        assert manager.expected_sequences["ETH-USD"] == 11
