"""
Test RSI analysis calculations.

Tests the RSI (Relative Strength Index) momentum indicator implementation
to ensure mathematical correctness.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
import pytest

from src.market.analysis.momentum_indicators import RSIAnalysis


class TestRSIAnalysis:
    """Test RSI calculation accuracy."""

    def test_rsi_with_known_values(self) -> None:
        """Test RSI calculation matches expected values."""
        # Example from Wilder's original RSI calculation
        # These are closing prices
        prices = [
            Decimal("44.34"),
            Decimal("44.09"),
            Decimal("44.15"),
            Decimal("43.61"),
            Decimal("44.33"),
            Decimal("44.83"),
            Decimal("45.10"),
            Decimal("45.42"),
            Decimal("45.84"),
            Decimal("46.08"),
            Decimal("45.89"),
            Decimal("46.03"),
            Decimal("45.61"),
            Decimal("46.28"),
            Decimal("46.28"),
        ]

        # Create price series as DataFrame
        df = pd.DataFrame(
            {
                "close": [float(p) for p in prices],
                "timestamp": pd.date_range(
                    start="2024-01-01", periods=len(prices), freq="D"
                ),
            }
        )
        df.set_index("timestamp", inplace=True)

        # Calculate RSI
        analysis = RSIAnalysis.from_price_series(
            prices=pd.Series([float(p) for p in prices]),
            symbol="TEST",
            timestamp=datetime.now(UTC),
            period=14,
        )

        # With 14-period RSI and this data, expected RSI should be around 70.53
        assert analysis.rsi_value == pytest.approx(70.53, abs=0.5)
        assert analysis.momentum_state == "bullish"
        assert analysis.is_overbought is True
        assert analysis.is_oversold is False

    def test_rsi_oversold_condition(self) -> None:
        """Test RSI correctly identifies oversold conditions."""
        # Create declining price series
        prices = [Decimal(str(50 - i * 2)) for i in range(15)]

        analysis = RSIAnalysis.from_price_series(
            prices=pd.Series([float(p) for p in prices]),
            symbol="TEST",
            timestamp=datetime.now(UTC),
            period=14,
        )

        # Strong downtrend should produce low RSI
        assert analysis.rsi_value < 30
        assert analysis.momentum_state in ["bearish", "strongly_bearish"]
        assert analysis.is_oversold is True
        assert analysis.is_overbought is False

    def test_rsi_neutral_condition(self) -> None:
        """Test RSI in sideways market."""
        # Create oscillating price series
        prices = []
        for i in range(20):
            if i % 2 == 0:
                prices.append(Decimal("50.00"))
            else:
                prices.append(Decimal("50.50"))

        analysis = RSIAnalysis.from_price_series(
            prices=pd.Series([float(p) for p in prices]),
            symbol="TEST",
            timestamp=datetime.now(UTC),
            period=14,
        )

        # Sideways market should produce RSI near 50
        assert 40 < analysis.rsi_value < 60
        assert analysis.momentum_state == "neutral"
        assert analysis.is_oversold is False
        assert analysis.is_overbought is False

    def test_rsi_insufficient_data(self) -> None:
        """Test RSI handles insufficient data gracefully."""
        # Less than period + 1 prices
        prices = [Decimal("50.00") for _ in range(10)]

        with pytest.raises(ValueError, match="Insufficient data"):
            RSIAnalysis.from_price_series(
                prices=pd.Series([float(p) for p in prices]),
                symbol="TEST",
                timestamp=datetime.now(UTC),
                period=14,
            )

    def test_rsi_divergence_detection(self) -> None:
        """Test RSI divergence detection."""
        # Price making higher highs but momentum weakening
        prices = [
            Decimal("50.00"),
            Decimal("51.00"),
            Decimal("52.00"),
            Decimal("53.00"),  # First peak
            Decimal("52.50"),
            Decimal("52.00"),
            Decimal("52.50"),
            Decimal("53.50"),  # Higher high in price
            Decimal("53.00"),
            Decimal("52.50"),
            Decimal("52.00"),
            Decimal("51.50"),
            Decimal("51.00"),
            Decimal("50.50"),
            Decimal("50.00"),
        ]

        analysis = RSIAnalysis.from_price_series(
            prices=pd.Series([float(p) for p in prices]),
            symbol="TEST",
            timestamp=datetime.now(UTC),
            period=14,
            check_divergence=True,
        )

        # This specific pattern might show bearish divergence
        # (price higher but RSI lower)
        assert hasattr(analysis, "divergence_type")

    def test_rsi_semantic_summary(self) -> None:
        """Test semantic summary generation for agents."""
        # Create bullish scenario
        prices = [Decimal(str(50 + i * 0.5)) for i in range(20)]

        analysis = RSIAnalysis.from_price_series(
            prices=pd.Series([float(p) for p in prices]),
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            period=14,
        )

        summary = analysis.semantic_summary()
        assert "RSI" in summary
        assert str(int(analysis.rsi_value)) in summary
        assert any(
            word in summary.lower()
            for word in ["bullish", "bearish", "neutral", "overbought", "oversold"]
        )

    def test_rsi_agent_context(self) -> None:
        """Test agent context generation."""
        prices = [Decimal(str(50 + i * 0.5)) for i in range(20)]

        analysis = RSIAnalysis.from_price_series(
            prices=pd.Series([float(p) for p in prices]),
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            period=14,
        )

        context = analysis.to_agent_context()

        # Check for the actual fields in our clean API
        assert context["indicator"] == "rsi"
        assert "value" in context
        assert "state" in context
        assert "strength" in context
        assert "key_levels" in context
        assert "signals" in context
        assert "interpretation" in context

        # Verify structure
        assert context["value"] == analysis.rsi_value
        assert context["state"] == analysis.momentum_state
        assert context["strength"] == analysis.momentum_strength

        # Check nested structures
        assert context["key_levels"]["current"] == analysis.rsi_value
        assert context["key_levels"]["overbought"] == 70
        assert context["key_levels"]["oversold"] == 30
        assert context["key_levels"]["neutral"] == 50

        assert context["signals"]["is_overbought"] == analysis.is_overbought
        assert context["signals"]["is_oversold"] == analysis.is_oversold

    def test_rsi_different_periods(self) -> None:
        """Test RSI with different period settings."""
        # Create price series with volatility but upward bias
        import math

        prices = []
        for i in range(30):
            # Upward trend with sine wave volatility
            base_price = 50 + i * 0.2
            volatility = math.sin(i * 0.5) * 2
            prices.append(Decimal(str(base_price + volatility)))

        price_series = pd.Series([float(p) for p in prices])

        # Test different periods
        rsi_9 = RSIAnalysis.from_price_series(
            prices=price_series,
            symbol="TEST",
            timestamp=datetime.now(UTC),
            period=9,
        )

        rsi_21 = RSIAnalysis.from_price_series(
            prices=price_series,
            symbol="TEST",
            timestamp=datetime.now(UTC),
            period=21,
        )

        # Both should show bullish bias but not be at 100
        assert 50 < rsi_9.rsi_value < 90
        assert 50 < rsi_21.rsi_value < 90
        # Shorter period RSI is more volatile, not necessarily higher
        assert abs(rsi_9.rsi_value - rsi_21.rsi_value) < 20

    def test_rsi_extreme_values(self) -> None:
        """Test RSI behavior at extremes."""
        # All gains (should approach 100)
        prices_up = [Decimal(str(50 + i)) for i in range(20)]
        analysis_up = RSIAnalysis.from_price_series(
            prices=pd.Series([float(p) for p in prices_up]),
            symbol="TEST",
            timestamp=datetime.now(UTC),
            period=14,
        )
        assert analysis_up.rsi_value > 90
        assert analysis_up.momentum_state == "strongly_bullish"

        # All losses (should approach 0)
        prices_down = [Decimal(str(70 - i)) for i in range(20)]
        analysis_down = RSIAnalysis.from_price_series(
            prices=pd.Series([float(p) for p in prices_down]),
            symbol="TEST",
            timestamp=datetime.now(UTC),
            period=14,
        )
        assert analysis_down.rsi_value < 10
        assert analysis_down.momentum_state == "strongly_bearish"

    def test_rsi_signal_generation(self) -> None:
        """Test RSI-based trading signal generation."""
        # Oversold condition should suggest potential buy
        prices = [Decimal(str(50 - i * 0.5)) for i in range(20)]
        analysis = RSIAnalysis.from_price_series(
            prices=pd.Series([float(p) for p in prices]),
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            period=14,
        )

        signal = analysis.suggest_signal()
        assert signal["bias"] == "bullish"  # Oversold = potential bounce
        assert signal["strength"] in ["weak", "moderate", "strong"]
        assert "RSI oversold" in signal["reason"]
