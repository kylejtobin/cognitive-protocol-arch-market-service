"""
Minimal tests to verify the core parsing functionality works.

These are the absolute minimum tests to prove our market service functions.
"""

from datetime import UTC
from decimal import Decimal

from src.market.adapters.coinbase.data import CoinbaseLevel2, TickerData
from src.market.model.snapshot import MarketSnapshot


def test_can_parse_ticker_data() -> None:
    """Verify we can parse ticker JSON into our model."""
    # Given: Realistic ticker JSON from Coinbase
    ticker_json = {
        "type": "ticker",
        "product_id": "BTC-USD",
        "price": "109473.61",
        "volume_24h": "1234.5678",
        "best_bid": "109473.61",
        "best_ask": "109486.48",
        "time": "2024-01-01T10:00:00Z",
    }

    # When: We parse it
    ticker = TickerData.model_validate(ticker_json)

    # Then: Basic fields work
    assert ticker.symbol == "BTC-USD"
    assert isinstance(ticker.price, Decimal)
    assert ticker.price == Decimal("109473.61")


def test_can_parse_order_book() -> None:
    """Verify we can parse level2 order book data."""
    # Given: Minimal order book snapshot
    level2_json = {
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
                        "price_level": "109473.61",
                        "new_quantity": "1.0",
                    },
                    {
                        "side": "offer",
                        "event_time": "2024-01-01T10:00:00Z",
                        "price_level": "109486.48",
                        "new_quantity": "1.0",
                    },
                ],
            }
        ],
    }

    # When: We parse it
    order_book = CoinbaseLevel2.model_validate(level2_json)

    # Then: Basic fields work
    assert order_book.symbol == "BTC-USD"
    assert order_book.best_bid == Decimal("109473.61")
    assert order_book.best_ask == Decimal("109486.48")
    assert order_book.spread == Decimal("12.87")


def test_can_create_market_snapshot() -> None:
    """Verify we can create a market snapshot with our components."""
    # Given: A ticker data from Coinbase
    ticker_data = TickerData.model_validate(
        {
            "type": "ticker",
            "product_id": "BTC-USD",
            "price": "109473.61",
            "time": "2024-01-01T10:00:00Z",
        }
    )

    # When: We create a snapshot with a domain model
    from datetime import datetime

    from src.market.model.ticker import MarketTicker

    # Convert to domain model
    ticker = MarketTicker(
        symbol=ticker_data.symbol,
        price=ticker_data.price,
        bid=ticker_data.bid,
        ask=ticker_data.ask,
        volume=ticker_data.volume,
        timestamp=ticker_data.timestamp,
    )

    snapshot = MarketSnapshot(
        symbol="BTC-USD",
        timestamp=datetime.now(UTC),
        ticker=ticker,
    )

    # Then: It works
    assert snapshot.symbol == "BTC-USD"
    assert snapshot.ticker is not None
    assert snapshot.ticker.price == Decimal("109473.61")
