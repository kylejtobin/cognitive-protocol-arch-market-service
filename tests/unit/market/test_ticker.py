"""
Tests for MarketTicker domain model with primitives.

Tests protocol satisfaction, domain operations, and primitive integration.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from src.market.domain.primitives import Percentage
from src.market.model.ticker import MarketTicker


class TestMarketTicker:
    """Test MarketTicker domain model."""

    def test_create_ticker_with_required_fields(self):
        """Test creating ticker with only required fields."""
        ticker = MarketTicker(
            symbol="BTC-USD",
            exchange="COINBASE",
            price=Decimal("50000"),
            size=Decimal("0.5"),
        )

        assert ticker.symbol == "BTC-USD"
        assert ticker.exchange == "COINBASE"
        assert ticker.price == Decimal("50000")
        assert ticker.size == Decimal("0.5")

        # Optional fields should be None
        assert ticker.bid is None
        assert ticker.ask is None
        assert ticker.volume is None

    def test_create_ticker_with_all_fields(self):
        """Test creating ticker with all fields."""
        ticker = MarketTicker(
            symbol="ETH-USD",
            exchange="BINANCE",
            price=Decimal("3000"),
            size=Decimal("10"),
            bid=Decimal("2999"),
            ask=Decimal("3001"),
            bid_size=Decimal("100"),
            ask_size=Decimal("150"),
            volume=Decimal("50000"),
            vwap=Decimal("3000.50"),
        )

        assert ticker.symbol == "ETH-USD"
        assert ticker.bid == Decimal("2999")
        assert ticker.ask == Decimal("3001")
        assert ticker.volume == Decimal("50000")

    def test_computed_primitives(self):
        """Test that computed primitives are accessible."""
        ticker = MarketTicker(
            symbol="BTC-USD",
            exchange="TEST",
            price=Decimal("50000"),
            size=Decimal("1"),
        )

        # Access computed properties
        price_prim = ticker.price_primitive
        assert price_prim.value == Decimal("50000")
        assert price_prim.format_display() == "$50,000.00"

        size_prim = ticker.size_primitive
        assert size_prim.value == Decimal("1")

    def test_spread_calculation(self):
        """Test spread calculation."""
        ticker = MarketTicker(
            symbol="BTC-USD",
            exchange="TEST",
            price=Decimal("50000"),
            size=Decimal("1"),
            bid=Decimal("49990"),
            ask=Decimal("50010"),
        )

        spread = ticker.spread  # Access as property, not method
        assert spread == Decimal("20")

    def test_spread_percentage(self):
        """Test spread percentage calculation."""
        ticker = MarketTicker(
            symbol="BTC-USD",
            exchange="TEST",
            price=Decimal("50000"),
            size=Decimal("1"),
            bid=Decimal("49990"),
            ask=Decimal("50010"),
        )

        spread_pct = ticker.spread_percentage()
        assert isinstance(spread_pct, Percentage)
        # Spread is 20, mid is 50000, so 20/50000 = 0.04%
        assert spread_pct.value == Decimal("0.04")

    def test_spread_basis_points(self):
        """Test spread calculation in basis points."""
        ticker = MarketTicker(
            symbol="BTC-USD",
            exchange="TEST",
            price=Decimal("50000"),
            size=Decimal("1"),
            bid=Decimal("49990"),
            ask=Decimal("50010"),
        )

        spread_bps = ticker.spread_basis_points()
        # 0.04% = 4 basis points
        assert spread_bps == 4

    def test_price_change_from(self):
        """Test price change calculation between tickers."""
        ticker1 = MarketTicker(
            symbol="BTC-USD",
            exchange="TEST",
            price=Decimal("50000"),
            size=Decimal("1"),
        )

        ticker2 = MarketTicker(
            symbol="BTC-USD",
            exchange="TEST",
            price=Decimal("52500"),
            size=Decimal("1"),
        )

        change = ticker2.price_change_from(ticker1)
        assert isinstance(change, Percentage)
        # (52500 - 50000) / 50000 = 5%
        assert change.value == Decimal("5")

    def test_is_liquid(self):
        """Test liquidity check."""
        ticker = MarketTicker(
            symbol="BTC-USD",
            exchange="TEST",
            price=Decimal("50000"),
            size=Decimal("1"),
            bid=Decimal("49995"),
            ask=Decimal("50005"),
            volume=Decimal("1000000"),
        )

        # Should be liquid with high volume and tight spread
        assert ticker.is_liquid(min_volume=Decimal("100000"), max_spread_bps=10) is True

        # Should not be liquid with stricter requirements
        assert (
            ticker.is_liquid(
                min_volume=Decimal("2000000"),  # Higher than actual
                max_spread_bps=1,  # Tighter than actual
            )
            is False
        )

    def test_format_summary(self):
        """Test human-readable summary formatting."""
        ticker = MarketTicker(
            symbol="BTC-USD",
            exchange="COINBASE",
            price=Decimal("50000"),
            size=Decimal("0.5"),
            bid=Decimal("49990"),
            ask=Decimal("50010"),
            volume=Decimal("1000000"),
        )

        summary = ticker.format_summary()
        assert "BTC-USD @ COINBASE" in summary
        assert "Price: $50,000.00" in summary
        assert "Bid/Ask: 49990/50010" in summary
        assert "Spread: 4bps" in summary
        assert "Volume:" in summary

    def test_protocol_satisfaction(self):
        """Test that model satisfies the protocol."""
        ticker = MarketTicker(
            symbol="TEST",
            exchange="TEST",
            price=Decimal("100"),
            size=Decimal("10"),
        )

        # These are the protocol requirements
        assert hasattr(ticker, "symbol")
        assert hasattr(ticker, "exchange")
        assert hasattr(ticker, "price")
        assert hasattr(ticker, "size")
        assert hasattr(ticker, "bid")
        assert hasattr(ticker, "ask")
        assert hasattr(ticker, "timestamp")

        # All values should be accessible
        assert ticker.symbol == "TEST"
        assert ticker.price == Decimal("100")
        assert ticker.bid is None  # Optional field

    def test_model_serialization(self):
        """Test model serialization to dict."""
        ticker = MarketTicker(
            symbol="BTC-USD",
            exchange="COINBASE",
            price=Decimal("50000"),
            size=Decimal("0.5"),
            bid=Decimal("49990"),
            ask=Decimal("50010"),
        )

        data = ticker.model_dump()
        assert data["symbol"] == "BTC-USD"
        assert data["exchange"] == "COINBASE"
        assert data["price"] == Decimal("50000")
        assert data["size"] == Decimal("0.5")
        assert data["bid"] == Decimal("49990")
        assert data["ask"] == Decimal("50010")

        # Computed fields should be excluded by default
        assert "price_primitive" not in data
        assert "size_primitive" not in data

        # Exclude None values
        data_exclude_none = ticker.model_dump(exclude_none=True)
        assert "volume" not in data_exclude_none
        assert "vwap" not in data_exclude_none

    def test_is_stale(self):
        """Test ticker staleness check."""
        # Create a fresh ticker
        fresh_ticker = MarketTicker(
            symbol="BTC-USD",
            exchange="TEST",
            price=Decimal("50000"),
            size=Decimal("1"),
        )

        # Should not be stale immediately
        assert fresh_ticker.is_stale(max_age_seconds=60) is False

        # Create a stale ticker with old timestamp
        old_timestamp = datetime.now(UTC) - timedelta(minutes=10)

        stale_ticker = MarketTicker(
            symbol="BTC-USD",
            exchange="TEST",
            price=Decimal("50000"),
            size=Decimal("1"),
            timestamp=old_timestamp,
        )

        # Should be stale after 5 minutes
        assert stale_ticker.is_stale(max_age_seconds=300) is True
        # But not stale if we allow 15 minutes
        assert stale_ticker.is_stale(max_age_seconds=900) is False

    def test_is_more_liquid_than(self):
        """Test liquidity comparison between tickers."""
        ticker1 = MarketTicker(
            symbol="BTC-USD",
            exchange="TEST",
            price=Decimal("50000"),
            size=Decimal("1"),
            volume=Decimal("1000000"),
        )

        ticker2 = MarketTicker(
            symbol="ETH-USD",
            exchange="TEST",
            price=Decimal("3000"),
            size=Decimal("10"),
            volume=Decimal("500000"),
        )

        assert ticker1.is_more_liquid_than(ticker2) is True
        assert ticker2.is_more_liquid_than(ticker1) is False

        # Test with None volumes
        ticker3 = MarketTicker(
            symbol="LTC-USD",
            exchange="TEST",
            price=Decimal("100"),
            size=Decimal("50"),
        )

        assert ticker1.is_more_liquid_than(ticker3) is False
        assert ticker3.is_more_liquid_than(ticker1) is False
