"""
Tests for domain primitives.

These tests verify the behavior of domain primitives including
conversions, operations, and protocol satisfaction.
"""

from decimal import Decimal

import pytest

from src.market.domain.primitives import (
    VWAP,
    BasisPoints,
    Percentage,
    Price,
    Size,
    Spread,
    Volume,
)


class TestPrice:
    """Test Price primitive behavior."""

    def test_price_creation(self) -> None:
        """Test basic price creation."""
        price = Price(value=Decimal("100.50"))
        assert price.value == Decimal("100.50")
        assert price.currency == "USD"

        price_eur = Price(value=Decimal("85.25"), currency="EUR")
        assert price_eur.currency == "EUR"

    def test_price_validation(self) -> None:
        """Test price validation."""
        # Negative price should fail
        with pytest.raises(ValueError):
            Price(value=Decimal("-10"))

    def test_price_precision(self) -> None:
        """Test price precision handling."""
        # Should be quantized to 8 decimal places
        price = Price(value=Decimal("100.123456789"))
        assert price.value == Decimal("100.12345679")

    def test_protocol_satisfaction(self) -> None:
        """Test conversion methods for protocol satisfaction."""
        price = Price(value=Decimal("100.50"))
        assert price.to_decimal() == Decimal("100.50")
        assert price.as_float() == 100.50

    def test_percentage_change(self) -> None:
        """Test percentage change calculation."""
        old_price = Price(value=Decimal("100"))
        new_price = Price(value=Decimal("110"))

        pct = new_price.percentage_change(old_price)
        assert pct.value == Decimal("10")  # 10% increase

        # Negative change
        pct2 = old_price.percentage_change(new_price)
        assert pct2.value == Decimal("-9.090909090909090909090909091")  # ~-9.09%

    def test_percentage_change_errors(self) -> None:
        """Test percentage change error cases."""
        price1 = Price(value=Decimal("100"))
        price2 = Price(value=Decimal("0"))
        price3 = Price(value=Decimal("100"), currency="EUR")

        # Zero price
        with pytest.raises(ValueError, match="zero price"):
            price1.percentage_change(price2)

        # Currency mismatch
        with pytest.raises(ValueError, match="Currency mismatch"):
            price1.percentage_change(price3)

    def test_with_percentage_change(self) -> None:
        """Test applying percentage change."""
        price = Price(value=Decimal("100"))
        pct = Percentage(value=Decimal("10"))  # 10%

        new_price = price.with_percentage_change(pct)
        assert new_price.value == Decimal("110")
        assert new_price.currency == price.currency

    def test_calculate_spread(self) -> None:
        """Test spread calculation."""
        bid = Price(value=Decimal("100"))
        ask = Price(value=Decimal("101"))

        spread = bid.calculate_spread_to(ask)
        assert spread.value == Decimal("1")
        assert spread.reference_price == bid

    def test_formatting(self) -> None:
        """Test price formatting."""
        price_usd = Price(value=Decimal("1234.5678"))
        assert price_usd.format_display() == "$1,234.57"
        assert price_usd.format_display(4) == "$1,234.5678"

        price_eur = Price(value=Decimal("1234.56"), currency="EUR")
        assert price_eur.format_display() == "1,234.56 EUR"


class TestSize:
    """Test Size primitive behavior."""

    def test_size_creation(self) -> None:
        """Test basic size creation."""
        size = Size(value=Decimal("10.5"))
        assert size.value == Decimal("10.5")

    def test_size_validation(self) -> None:
        """Test size validation."""
        with pytest.raises(ValueError):
            Size(value=Decimal("-1"))

    def test_size_operations(self) -> None:
        """Test size arithmetic operations."""
        size1 = Size(value=Decimal("10"))
        size2 = Size(value=Decimal("5"))

        # Addition
        size3 = size1 + size2
        assert size3.value == Decimal("15")

        # Subtraction
        size4 = size1 - size2
        assert size4.value == Decimal("5")

        # Multiplication
        size5 = size1 * 2
        assert size5.value == Decimal("20")

        size6 = size1 * Decimal("1.5")
        assert size6.value == Decimal("15")

    def test_size_subtraction_error(self) -> None:
        """Test subtraction error when result would be negative."""
        size1 = Size(value=Decimal("5"))
        size2 = Size(value=Decimal("10"))

        with pytest.raises(ValueError, match="Cannot subtract larger"):
            size1 - size2

    def test_is_zero(self) -> None:
        """Test zero check."""
        zero_size = Size(value=Decimal("0"))
        assert zero_size.is_zero()

        non_zero = Size(value=Decimal("0.001"))
        assert not non_zero.is_zero()


class TestPercentageAndBasisPoints:
    """Test Percentage and BasisPoints primitives."""

    def test_percentage_creation(self) -> None:
        """Test percentage creation."""
        pct = Percentage(value=Decimal("50"))  # 50%
        assert pct.value == Decimal("50")

    def test_percentage_conversions(self) -> None:
        """Test percentage conversions."""
        pct = Percentage(value=Decimal("50"))  # 50%

        # To decimal
        assert pct.as_decimal() == Decimal("0.5")

        # To basis points
        bps = pct.as_basis_points()
        assert bps.value == Decimal("5000")  # 5000 bps

    def test_basis_points_conversions(self) -> None:
        """Test basis points conversions."""
        bps = BasisPoints(value=Decimal("100"))  # 100 bps = 1%

        # To percentage
        pct = bps.to_percentage()
        assert pct.value == Decimal("1")  # 1%

        # To decimal
        assert bps.as_decimal() == Decimal("0.01")

    def test_formatting(self) -> None:
        """Test formatting."""
        pct = Percentage(value=Decimal("12.345"))
        assert pct.format_display() == "12.34%"  # Note: rounds down
        assert pct.format_display(1) == "12.3%"

        bps = BasisPoints(value=Decimal("150.5"))
        assert bps.format_display() == "150.5bps"


class TestSpread:
    """Test Spread primitive behavior."""

    def test_spread_creation(self) -> None:
        """Test spread creation."""
        ref_price = Price(value=Decimal("100"))
        spread = Spread(value=Decimal("1"), reference_price=ref_price)
        assert spread.value == Decimal("1")
        assert spread.reference_price == ref_price

    def test_spread_conversions(self) -> None:
        """Test spread conversions."""
        ref_price = Price(value=Decimal("100"))
        spread = Spread(value=Decimal("1"), reference_price=ref_price)

        # To percentage
        pct = spread.as_percentage()
        assert pct.value == Decimal("1")  # 1%

        # To basis points
        bps = spread.as_basis_points()
        assert bps.value == Decimal("100")  # 100 bps

    def test_spread_zero_reference(self) -> None:
        """Test spread with zero reference price."""
        ref_price = Price(value=Decimal("0"))
        spread = Spread(value=Decimal("1"), reference_price=ref_price)

        pct = spread.as_percentage()
        assert pct.value == Decimal("0")

    def test_is_tight(self) -> None:
        """Test tight spread check."""
        ref_price = Price(value=Decimal("1000"))
        tight_spread = Spread(value=Decimal("0.1"), reference_price=ref_price)
        wide_spread = Spread(value=Decimal("1.5"), reference_price=ref_price)

        assert tight_spread.is_tight()  # 0.1/1000 = 0.01% = 1 bps
        assert not wide_spread.is_tight()  # 1.5/1000 = 0.15% = 15 bps

        # Edge case - exactly at threshold
        edge_spread = Spread(value=Decimal("1"), reference_price=ref_price)
        assert edge_spread.is_tight()  # 1/1000 = 0.1% = 10 bps (exactly at default)

        # Custom threshold
        assert wide_spread.is_tight(max_bps=Decimal("20"))


class TestVolume:
    """Test Volume primitive behavior."""

    def test_volume_creation(self) -> None:
        """Test volume creation."""
        size = Size(value=Decimal("1000"))
        volume = Volume(size=size)
        assert volume.size == size
        assert volume.timeframe_hours == 24

        volume_1h = Volume(size=size, timeframe_hours=1)
        assert volume_1h.timeframe_hours == 1

    def test_daily_equivalent(self) -> None:
        """Test daily volume calculation."""
        size = Size(value=Decimal("100"))

        # 24h volume is already daily
        volume_24h = Volume(size=size, timeframe_hours=24)
        assert volume_24h.daily_equivalent() == size

        # 1h volume extrapolated to 24h
        volume_1h = Volume(size=size, timeframe_hours=1)
        daily = volume_1h.daily_equivalent()
        assert daily.value == Decimal("2400")

    def test_hourly_rate(self) -> None:
        """Test hourly rate calculation."""
        size = Size(value=Decimal("2400"))
        volume = Volume(size=size, timeframe_hours=24)

        hourly = volume.hourly_rate()
        assert hourly.value == Decimal("100")


class TestVWAP:
    """Test VWAP primitive behavior."""

    def test_vwap_creation(self) -> None:
        """Test VWAP creation."""
        price = Price(value=Decimal("100"))
        volume = Volume(size=Size(value=Decimal("1000")))
        vwap = VWAP(price=price, volume=volume)

        assert vwap.price == price
        assert vwap.volume == volume
        assert vwap.confidence == Decimal("1.0")

    def test_vwap_calculation(self) -> None:
        """Test VWAP calculation."""
        prices = [
            Price(value=Decimal("100")),
            Price(value=Decimal("101")),
            Price(value=Decimal("102")),
        ]
        sizes = [
            Size(value=Decimal("10")),
            Size(value=Decimal("20")),
            Size(value=Decimal("30")),
        ]

        vwap = VWAP.calculate(prices, sizes)

        # Manual calculation: (100*10 + 101*20 + 102*30) / (10+20+30) = 101.333...
        expected = Decimal("101.33333333")
        assert vwap.price.value == expected
        assert vwap.volume.size.value == Decimal("60")
        assert vwap.confidence == Decimal("0.03")  # 3/100

    def test_vwap_calculation_errors(self) -> None:
        """Test VWAP calculation error cases."""
        # Empty data
        with pytest.raises(ValueError, match="empty data"):
            VWAP.calculate([], [])

        # Mismatched lengths
        prices = [Price(value=Decimal("100"))]
        sizes = [Size(value=Decimal("10")), Size(value=Decimal("20"))]
        with pytest.raises(ValueError, match="same length"):
            VWAP.calculate(prices, sizes)

        # Multiple currencies
        prices_multi = [
            Price(value=Decimal("100"), currency="USD"),
            Price(value=Decimal("100"), currency="EUR"),
        ]
        sizes = [Size(value=Decimal("10")), Size(value=Decimal("10"))]
        with pytest.raises(ValueError, match="Multiple currencies"):
            VWAP.calculate(prices_multi, sizes)

        # Zero volume
        prices = [Price(value=Decimal("100"))]
        sizes = [Size(value=Decimal("0"))]
        with pytest.raises(ValueError, match="zero total size"):
            VWAP.calculate(prices, sizes)

    def test_is_reliable(self) -> None:
        """Test reliability check."""
        price = Price(value=Decimal("100"))
        volume = Volume(size=Size(value=Decimal("1000")))

        high_conf = VWAP(price=price, volume=volume, confidence=Decimal("0.9"))
        low_conf = VWAP(price=price, volume=volume, confidence=Decimal("0.5"))

        assert high_conf.is_reliable()
        assert not low_conf.is_reliable()
        assert low_conf.is_reliable(min_confidence=Decimal("0.4"))
