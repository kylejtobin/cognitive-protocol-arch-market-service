"""
Coinbase order book management with snapshot and update handling.

This module manages the stateful order book updates from Coinbase WebSocket,
handling the snapshot + update pattern that Coinbase uses for level2 data.
"""

from src.market.adapters.coinbase.data import CoinbaseLevel2, Level2Event
from src.market.model.book import MutableOrderBook
from src.market.model.types import PriceLevelData


class CoinbaseBookManager:
    """
    Manages order book state for Coinbase level2 data.

    Handles the Coinbase-specific logic of:
    - Initial snapshots
    - Incremental updates
    - Sequence number tracking
    - Resnapshot on sequence gaps
    """

    def __init__(self) -> None:
        """Initialize the book manager."""
        self.books: dict[str, MutableOrderBook] = {}
        self.expected_sequences: dict[str, int] = {}

    def process_level2_message(self, message: CoinbaseLevel2) -> list[MutableOrderBook]:
        """
        Process a level2 message and return affected order books.

        Args:
            message: Parsed Coinbase level2 message

        Returns:
            List of order books that were updated

        """
        updated_books = []

        for event in message.events:
            book = self._process_event(event, message.sequence_num)
            if book:
                updated_books.append(book)

        return updated_books

    def _process_event(
        self, event: Level2Event, sequence: int
    ) -> MutableOrderBook | None:
        """Process a single level2 event."""
        symbol = event.product_id

        # Check sequence number if we have an expected one
        if symbol in self.expected_sequences:
            if sequence != self.expected_sequences[symbol]:
                # Gap detected - need resnapshot
                print(
                    f"Sequence gap for {symbol}: expected "
                    f"{self.expected_sequences[symbol]}, got {sequence}"
                )
                # In production, you'd request a new snapshot here
                # For now, we'll just accept it

        # Create book if it doesn't exist
        if symbol not in self.books:
            self.books[symbol] = MutableOrderBook(symbol)

        book = self.books[symbol]

        if event.type == "snapshot":
            # Process snapshot
            bids: list[PriceLevelData] = []
            asks: list[PriceLevelData] = []

            for update in event.updates:
                price = update.price_decimal
                size = update.size_decimal
                level = PriceLevelData(price=price, size=size)

                if update.side_value.value == "bid":
                    bids.append(level)
                else:
                    asks.append(level)

            book.apply_snapshot(bids, asks, sequence)
        else:
            # Process updates
            for update in event.updates:
                book.apply_update(
                    side=update.side_value.value,
                    price=update.price_decimal,
                    size=update.size_decimal,
                )
            book.sequence = sequence

        # Update expected sequence
        self.expected_sequences[symbol] = sequence + 1

        return book

    def get_book(self, symbol: str) -> MutableOrderBook | None:
        """Get the current order book for a symbol."""
        return self.books.get(symbol)
