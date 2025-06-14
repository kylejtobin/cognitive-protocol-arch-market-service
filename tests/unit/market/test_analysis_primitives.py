"""
Tests for analysis domain primitives.

Tests RSIValue, MACDValue, StochasticValue, VolumeProfile, and BollingerBandPosition.
"""

from decimal import Decimal

import pytest

from src.market.domain.analysis_primitives import (
    BollingerBandPosition,
    MACDValue,
    RSIValue,
    StochasticValue,
    VolumeProfile,
)


class TestRSIValue:
    """Test RSIValue domain primitive."""

    def test_create_valid_rsi(self) -> None:
        """Test creating valid RSI values."""
        rsi = RSIValue(value=Decimal("50"))
        assert rsi.value == Decimal("50")
        assert rsi.zone == "neutral"
        assert rsi.strength == "weak"

    def test_rsi_zones(self) -> None:
        """Test RSI zone detection."""
        oversold = RSIValue(value=Decimal("25"))
        assert oversold.zone == "oversold"

        neutral = RSIValue(value=Decimal("50"))
        assert neutral.zone == "neutral"

        overbought = RSIValue(value=Decimal("75"))
        assert overbought.zone == "overbought"

    def test_rsi_strength(self) -> None:
        """Test RSI strength classification."""
        extreme_low = RSIValue(value=Decimal("15"))
        assert extreme_low.strength == "extreme"

        strong = RSIValue(value=Decimal("25"))
        assert strong.strength == "strong"

        moderate = RSIValue(value=Decimal("35"))
        assert moderate.strength == "moderate"

        weak = RSIValue(value=Decimal("50"))
        assert weak.strength == "weak"

    def test_rsi_validation(self) -> None:
        """Test RSI value validation."""
        with pytest.raises(ValueError):
            RSIValue(value=Decimal("-1"))

        with pytest.raises(ValueError):
            RSIValue(value=Decimal("101"))

    def test_distance_calculations(self) -> None:
        """Test distance calculation methods."""
        rsi = RSIValue(value=Decimal("65"))

        # Distance from neutral
        assert rsi.distance_from_neutral() == Decimal("15")

        # Distance from zone boundary
        assert rsi.distance_from_zone_boundary() == Decimal("5")  # 5 away from 70

    def test_format_display(self) -> None:
        """Test display formatting."""
        oversold = RSIValue(value=Decimal("25.5"))
        assert oversold.format_display() == "RSI 25.5 ↓"

        overbought = RSIValue(value=Decimal("80.2"))
        assert overbought.format_display() == "RSI 80.2 ↑"


class TestMACDValue:
    """Test MACDValue domain primitive."""

    def test_create_macd(self) -> None:
        """Test creating MACD value."""
        macd = MACDValue(macd_line=Decimal("10"), signal_line=Decimal("8"))
        assert macd.macd_line == Decimal("10")
        assert macd.signal_line == Decimal("8")
        assert macd.histogram == Decimal("2")

    def test_trend_detection(self) -> None:
        """Test trend detection."""
        bullish = MACDValue(macd_line=Decimal("10"), signal_line=Decimal("8"))
        assert bullish.trend == "bullish"
        assert bullish.is_positive is True

        bearish = MACDValue(macd_line=Decimal("-10"), signal_line=Decimal("-8"))
        assert bearish.trend == "bearish"
        assert bearish.is_positive is False

        neutral = MACDValue(macd_line=Decimal("0.2"), signal_line=Decimal("0.1"))
        assert neutral.trend == "neutral"

    def test_crossover_strength(self) -> None:
        """Test crossover strength calculation."""
        strong_cross = MACDValue(macd_line=Decimal("10"), signal_line=Decimal("5"))
        strength = strong_cross.crossover_strength()
        assert strength.value == Decimal("100")  # (5/5) * 100

        weak_cross = MACDValue(macd_line=Decimal("10"), signal_line=Decimal("9"))
        strength = weak_cross.crossover_strength()
        assert abs(strength.value - Decimal("11.11")) < 0.01  # (1/9) * 100

    def test_momentum_direction(self) -> None:
        """Test momentum direction detection."""
        accelerating = MACDValue(macd_line=Decimal("10"), signal_line=Decimal("5"))
        assert accelerating.momentum_direction() == "accelerating"

        decelerating = MACDValue(macd_line=Decimal("5"), signal_line=Decimal("10"))
        assert decelerating.momentum_direction() == "decelerating"

        stable = MACDValue(macd_line=Decimal("5"), signal_line=Decimal("5.05"))
        assert stable.momentum_direction() == "stable"

    def test_format_display(self) -> None:
        """Test display formatting."""
        macd = MACDValue(macd_line=Decimal("12.50"), signal_line=Decimal("10.25"))
        assert macd.format_display() == "MACD 12.50/10.25 ↑"


class TestStochasticValue:
    """Test StochasticValue domain primitive."""

    def test_create_stochastic(self) -> None:
        """Test creating stochastic value."""
        stoch = StochasticValue(k_value=Decimal("80"), d_value=Decimal("75"))
        assert stoch.k_value == Decimal("80")
        assert stoch.d_value == Decimal("75")
        assert stoch.zone == "overbought"

    def test_zones(self) -> None:
        """Test zone detection."""
        oversold = StochasticValue(k_value=Decimal("15"), d_value=Decimal("20"))
        assert oversold.zone == "oversold"

        neutral = StochasticValue(k_value=Decimal("50"), d_value=Decimal("50"))
        assert neutral.zone == "neutral"

        overbought = StochasticValue(k_value=Decimal("85"), d_value=Decimal("80"))
        assert overbought.zone == "overbought"

    def test_momentum(self) -> None:
        """Test momentum detection."""
        increasing = StochasticValue(k_value=Decimal("60"), d_value=Decimal("50"))
        assert increasing.momentum == "increasing"

        decreasing = StochasticValue(k_value=Decimal("40"), d_value=Decimal("50"))
        assert decreasing.momentum == "decreasing"

        stable = StochasticValue(k_value=Decimal("50"), d_value=Decimal("51"))
        assert stable.momentum == "stable"

    def test_divergence(self) -> None:
        """Test divergence calculation."""
        stoch = StochasticValue(k_value=Decimal("60"), d_value=Decimal("50"))
        assert stoch.divergence() == Decimal("10")

    def test_crossing_detection(self) -> None:
        """Test crossing detection."""
        crossing = StochasticValue(k_value=Decimal("50"), d_value=Decimal("50.5"))
        assert crossing.is_crossing() is True

        not_crossing = StochasticValue(k_value=Decimal("60"), d_value=Decimal("50"))
        assert not_crossing.is_crossing() is False

    def test_format_display(self) -> None:
        """Test display formatting."""
        stoch = StochasticValue(k_value=Decimal("82.5"), d_value=Decimal("78.3"))
        assert stoch.format_display() == "Stoch %K:82.5/%D:78.3 ↑"


class TestVolumeProfile:
    """Test VolumeProfile domain primitive."""

    def test_create_volume_profile(self) -> None:
        """Test creating volume profile."""
        profile = VolumeProfile(
            current_volume=Decimal("1000000"),
            average_volume=Decimal("500000"),
            buy_volume=Decimal("600000"),
            sell_volume=Decimal("400000"),
        )
        assert profile.current_volume == Decimal("1000000")
        assert profile.volume_ratio == Decimal("2")

    def test_pressure_calculations(self) -> None:
        """Test buy/sell pressure calculations."""
        profile = VolumeProfile(
            current_volume=Decimal("1000000"),
            average_volume=Decimal("500000"),
            buy_volume=Decimal("700000"),
            sell_volume=Decimal("300000"),
        )
        assert profile.buy_pressure.value == Decimal("70")
        assert profile.sell_pressure.value == Decimal("30")
        assert profile.pressure_balance == "buy_heavy"

    def test_balanced_pressure(self) -> None:
        """Test balanced pressure detection."""
        profile = VolumeProfile(
            current_volume=Decimal("1000000"),
            average_volume=Decimal("500000"),
            buy_volume=Decimal("500000"),
            sell_volume=Decimal("500000"),
        )
        assert profile.pressure_balance == "balanced"

    def test_abnormal_volume(self) -> None:
        """Test abnormal volume detection."""
        high_volume = VolumeProfile(
            current_volume=Decimal("3000000"),
            average_volume=Decimal("1000000"),
            buy_volume=Decimal("2000000"),
            sell_volume=Decimal("1000000"),
        )
        assert high_volume.is_abnormal() is True
        assert high_volume.is_abnormal(threshold=Decimal("4")) is False

    def test_climactic_volume(self) -> None:
        """Test climactic volume detection."""
        climactic = VolumeProfile(
            current_volume=Decimal("5000000"),
            average_volume=Decimal("1000000"),
            buy_volume=Decimal("4500000"),
            sell_volume=Decimal("500000"),
        )
        assert climactic.is_climactic() is True

        not_climactic = VolumeProfile(
            current_volume=Decimal("2000000"),
            average_volume=Decimal("1000000"),
            buy_volume=Decimal("1100000"),
            sell_volume=Decimal("900000"),
        )
        assert not_climactic.is_climactic() is False

    def test_format_display(self) -> None:
        """Test display formatting."""
        profile = VolumeProfile(
            current_volume=Decimal("2500000"),
            average_volume=Decimal("1000000"),
            buy_volume=Decimal("1800000"),
            sell_volume=Decimal("700000"),
        )
        assert profile.format_display() == "Vol 2.5x avg (HIGH) ↑"


class TestBollingerBandPosition:
    """Test BollingerBandPosition domain primitive."""

    def test_create_bb_position(self) -> None:
        """Test creating Bollinger Band position."""
        bb = BollingerBandPosition(
            price=Decimal("100"),
            upper_band=Decimal("110"),
            middle_band=Decimal("100"),
            lower_band=Decimal("90"),
        )
        assert bb.price == Decimal("100")
        assert bb.band_width == Decimal("20")

    def test_position_percentage(self) -> None:
        """Test position percentage calculation."""
        # Price at middle
        bb_middle = BollingerBandPosition(
            price=Decimal("100"),
            upper_band=Decimal("110"),
            middle_band=Decimal("100"),
            lower_band=Decimal("90"),
        )
        assert bb_middle.position_pct.value == Decimal("50")

        # Price at upper
        bb_upper = BollingerBandPosition(
            price=Decimal("110"),
            upper_band=Decimal("110"),
            middle_band=Decimal("100"),
            lower_band=Decimal("90"),
        )
        assert bb_upper.position_pct.value == Decimal("100")

        # Price outside bands
        bb_outside = BollingerBandPosition(
            price=Decimal("115"),
            upper_band=Decimal("110"),
            middle_band=Decimal("100"),
            lower_band=Decimal("90"),
        )
        assert bb_outside.position_pct.value == Decimal("125")

    def test_position_zones(self) -> None:
        """Test position zone detection."""
        above = BollingerBandPosition(
            price=Decimal("115"),
            upper_band=Decimal("110"),
            middle_band=Decimal("100"),
            lower_band=Decimal("90"),
        )
        assert above.position_zone == "above_upper"

        upper_half = BollingerBandPosition(
            price=Decimal("105"),
            upper_band=Decimal("110"),
            middle_band=Decimal("100"),
            lower_band=Decimal("90"),
        )
        assert upper_half.position_zone == "upper_half"

    def test_squeeze_detection(self) -> None:
        """Test Bollinger Band squeeze detection."""
        squeeze = BollingerBandPosition(
            price=Decimal("100"),
            upper_band=Decimal("101"),
            middle_band=Decimal("100"),
            lower_band=Decimal("99"),
        )
        assert squeeze.squeeze_detected is True
        assert squeeze.band_width_pct.value == Decimal("2")

        no_squeeze = BollingerBandPosition(
            price=Decimal("100"),
            upper_band=Decimal("105"),
            middle_band=Decimal("100"),
            lower_band=Decimal("95"),
        )
        assert no_squeeze.squeeze_detected is False

    def test_distance_from_band(self) -> None:
        """Test distance calculation from bands."""
        bb = BollingerBandPosition(
            price=Decimal("105"),
            upper_band=Decimal("110"),
            middle_band=Decimal("100"),
            lower_band=Decimal("90"),
        )
        assert bb.distance_from_band("upper") == Decimal("5")
        assert bb.distance_from_band("middle") == Decimal("5")
        assert bb.distance_from_band("lower") == Decimal("15")

    def test_format_display(self) -> None:
        """Test display formatting."""
        bb = BollingerBandPosition(
            price=Decimal("108"),
            upper_band=Decimal("110"),
            middle_band=Decimal("100"),
            lower_band=Decimal("90"),
        )
        assert bb.format_display() == "BB 90% ↑"

        squeeze_bb = BollingerBandPosition(
            price=Decimal("100"),
            upper_band=Decimal("101"),
            middle_band=Decimal("100"),
            lower_band=Decimal("99"),
        )
        assert squeeze_bb.format_display() == "BB 50% ↑ [SQUEEZE]"
