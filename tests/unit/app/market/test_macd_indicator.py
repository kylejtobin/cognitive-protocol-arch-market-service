"""
Tests for MACD (Moving Average Convergence Divergence) indicator.

Tests MACD calculations, crossover detection, and divergence analysis.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.market.analysis.macd import MACDAnalysis


class TestMACDAnalysis:
    """Test MACD indicator functionality."""

    def test_basic_macd_calculation(self) -> None:
        """Test basic MACD calculation with known values."""
        # Create a simple price series
        prices = pd.Series(
            [
                100,
                101,
                102,
                101,
                103,
                105,
                104,
                106,
                108,
                107,
                109,
                111,
                110,
                112,
                114,
                113,
                115,
                117,
                116,
                118,
                120,
                119,
                121,
                123,
                122,
                124,
                126,
                125,
                127,
                129,
                128,
                130,
                132,
                131,
                133,
                135,
            ]
        )

        macd = MACDAnalysis.from_price_series(
            prices=prices,
            symbol="TEST",
            timestamp=datetime.now(UTC),
            fast_period=12,
            slow_period=26,
            signal_period=9,
        )

        # Basic validations
        assert macd.symbol == "TEST"
        assert macd.fast_period == 12
        assert macd.slow_period == 26
        assert macd.signal_period == 9

        # MACD values should be calculated
        assert isinstance(macd.macd_value, float)
        assert isinstance(macd.signal_value, float)
        assert isinstance(macd.histogram_value, float)

        # Histogram = MACD - Signal
        assert (
            abs(macd.histogram_value - (macd.macd_value - macd.signal_value)) < 0.0001
        )

    def test_insufficient_data_error(self) -> None:
        """Test error when insufficient data provided."""
        prices = pd.Series([100, 101, 102])  # Only 3 prices

        with pytest.raises(ValueError, match="Insufficient data"):
            MACDAnalysis.from_price_series(
                prices=prices,
                symbol="TEST",
                timestamp=datetime.now(UTC),
            )

    def test_bullish_crossover_detection(self) -> None:
        """Test detection of bullish crossover."""
        # Create price series with upward trend
        prices = pd.Series([100 + i * 0.5 + np.sin(i / 5) * 2 for i in range(50)])

        macd = MACDAnalysis.from_price_series(
            prices=prices,
            symbol="TEST",
            timestamp=datetime.now(UTC),
        )

        # For a generally upward trend, we expect bullish state
        assert macd.trend in ["bullish", "neutral"]

    def test_bearish_crossover_detection(self) -> None:
        """Test detection of bearish crossover."""
        # Create price series with downward trend
        prices = pd.Series([150 - i * 0.5 + np.sin(i / 5) * 2 for i in range(50)])

        macd = MACDAnalysis.from_price_series(
            prices=prices,
            symbol="TEST",
            timestamp=datetime.now(UTC),
        )

        # For a generally downward trend, we expect bearish state
        assert macd.trend in ["bearish", "neutral"]

    def test_trend_strength_classification(self) -> None:
        """Test trend strength classification."""
        # Strong trend
        strong_trend = pd.Series([100 + i * 2 for i in range(50)])
        macd_strong = MACDAnalysis.from_price_series(
            prices=strong_trend,
            symbol="TEST",
            timestamp=datetime.now(UTC),
        )

        # Weak/sideways trend
        weak_trend = pd.Series([100 + np.sin(i / 5) * 0.5 for i in range(50)])
        macd_weak = MACDAnalysis.from_price_series(
            prices=weak_trend,
            symbol="TEST",
            timestamp=datetime.now(UTC),
        )

        # Strong trend should have larger histogram values on average
        assert abs(macd_strong.histogram_value) >= abs(macd_weak.histogram_value)

    def test_semantic_summary(self) -> None:
        """Test semantic summary generation."""
        prices = pd.Series([100 + i * 0.5 for i in range(50)])

        macd = MACDAnalysis.from_price_series(
            prices=prices,
            symbol="TEST",
            timestamp=datetime.now(UTC),
        )

        summary = macd.semantic_summary()
        assert isinstance(summary, str)
        assert "MACD" in summary
        assert macd.trend in summary.lower()

    def test_to_agent_context(self) -> None:
        """Test agent context generation."""
        prices = pd.Series([100 + i * 0.5 for i in range(50)])

        macd = MACDAnalysis.from_price_series(
            prices=prices,
            symbol="TEST",
            timestamp=datetime.now(UTC),
        )

        context = macd.to_agent_context()

        # Check that we get a typed MACDAgentContext object
        from src.market.analysis.contexts import MACDAgentContext

        assert isinstance(context, MACDAgentContext)

        # Check structure
        assert context.indicator == "macd"
        assert context.trend == macd.trend
        assert context.momentum == macd.momentum

        # Check values (exact match since they come from same source)
        assert context.values.macd == macd.macd_value
        assert context.values.signal == macd.signal_value
        assert context.values.histogram == macd.histogram_value

        # Check signals
        assert context.signals.histogram_trend in [
            "expanding",
            "contracting",
            "neutral",
        ]
        assert context.signals.signal_position in [
            "above_signal",
            "below_signal",
            "at_signal",
        ]
        assert context.signals.zero_cross in ["bullish", "bearish", "none"]

    def test_suggest_signal(self) -> None:
        """Test signal suggestion generation."""
        prices = pd.Series([100 + i * 0.5 for i in range(50)])

        macd = MACDAnalysis.from_price_series(
            prices=prices,
            symbol="TEST",
            timestamp=datetime.now(UTC),
        )

        signal = macd.suggest_signal()

        # Check that we get a typed SignalSuggestion object
        from src.market.analysis.contexts import SignalSuggestion

        assert isinstance(signal, SignalSuggestion)

        # Check valid values
        assert signal.bias in ["bullish", "bearish", "neutral"]
        assert signal.strength in ["strong", "moderate", "weak"]
        assert signal.reason
        assert signal.action

    def test_field_validation(self) -> None:
        """Test field validation for periods."""
        prices = pd.Series([100 + i for i in range(50)])

        # Test that slow period must be greater than fast period
        with pytest.raises(
            ValueError, match="Slow period must be greater than fast period"
        ):
            MACDAnalysis.from_price_series(
                prices=prices,
                symbol="TEST",
                timestamp=datetime.now(UTC),
                fast_period=26,  # Fast is greater than slow
                slow_period=12,
                signal_period=9,
            )

    def test_crossover_in_signal(self) -> None:
        """Test that crossover affects signal suggestion."""
        # Create a series that will likely have a crossover
        prices: list[float] = []
        for i in range(40):
            if i < 20:
                prices.append(100 - i * 0.5)  # Downtrend
            else:
                prices.append(90 + (i - 20) * 1.0)  # Uptrend

        prices_series = pd.Series(prices)

        macd = MACDAnalysis.from_price_series(
            prices=prices_series,
            symbol="TEST",
            timestamp=datetime.now(UTC),
        )

        signal = macd.suggest_signal()

        # If crossover detected, it should be mentioned in the reason
        if macd.crossover_detected:
            assert "crossover" in signal.reason.lower()

    def test_edge_cases(self) -> None:
        """Test edge cases like flat prices."""
        # Flat prices
        flat_prices = pd.Series([100.0] * 50)

        macd_flat = MACDAnalysis.from_price_series(
            prices=flat_prices,
            symbol="TEST",
            timestamp=datetime.now(UTC),
        )

        # With flat prices, MACD should be near zero
        assert abs(macd_flat.macd_value) < 0.1
        assert abs(macd_flat.signal_value) < 0.1
        assert abs(macd_flat.histogram_value) < 0.1
        assert macd_flat.trend == "neutral"

    def test_registry_integration(self) -> None:
        """Test that MACD is properly registered."""
        from src.market.analysis.registry import IndicatorRegistry

        # Create isolated test registry
        test_registry = IndicatorRegistry.testing()

        # Import the MACDAnalysis class
        from src.market.analysis.macd import MACDAnalysis

        # Manually register it
        test_registry.indicators["macd"] = MACDAnalysis

        # Check registration
        assert test_registry.has("macd")

        # Get the class and use its from_price_series method
        prices = pd.Series([100 + i for i in range(50)])
        macd_class = test_registry.get("macd")

        assert macd_class is MACDAnalysis

        # Use the class method to create instance
        macd = MACDAnalysis.from_price_series(
            prices=prices,
            symbol="TEST",
            timestamp=datetime.now(UTC),
            fast_period=12,
            slow_period=26,
            signal_period=9,
        )

        assert isinstance(macd, macd_class)
        assert macd.symbol == "TEST"
