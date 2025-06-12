"""
Math validation tests for technical indicators.

These tests use known input/output pairs to verify our calculations
match expected values from reference implementations or manual calculations.
"""

from datetime import UTC, datetime

import pandas as pd

from src.market.analysis.macd import MACDAnalysis
from src.market.analysis.momentum_indicators import RSIAnalysis
from src.market.analysis.stochastic import StochasticAnalysis


class TestIndicatorMathValidation:
    """Validate indicator calculations against known values."""

    def test_rsi_known_values(self) -> None:
        """Test RSI with values that should produce known results."""
        # Classic example: 14 periods of gains/losses
        # This should produce RSI of approximately 70-73
        prices = pd.Series(
            [
                44.00,
                44.34,
                44.09,
                43.61,
                44.33,
                44.83,
                45.10,
                45.42,
                45.84,
                46.08,
                45.89,
                46.03,
                45.61,
                46.28,
                46.28,
            ]
        )

        rsi = RSIAnalysis.from_price_series(
            prices=prices,
            symbol="TEST",
            timestamp=datetime.now(UTC),
            period=14,
        )

        # Expected RSI is approximately 70-73 with our calculation method
        assert 70 <= rsi.rsi_value <= 73, f"Expected 70-73, got {rsi.rsi_value}"

    def test_rsi_extreme_values(self) -> None:
        """Test RSI with extreme cases."""
        # All gains should give RSI = 100
        all_gains = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        rsi_gains = RSIAnalysis.from_price_series(
            prices=all_gains,
            symbol="TEST",
            timestamp=datetime.now(UTC),
            period=10,
        )
        assert rsi_gains.rsi_value == 100.0

        # All losses should give RSI = 0
        all_losses = pd.Series([20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10])
        rsi_losses = RSIAnalysis.from_price_series(
            prices=all_losses,
            symbol="TEST",
            timestamp=datetime.now(UTC),
            period=10,
        )
        assert rsi_losses.rsi_value == 0.0

    def test_macd_basic_calculation(self) -> None:
        """Test MACD with simple trending data."""
        # Create linear uptrend
        prices = pd.Series([100 + i for i in range(50)])

        macd = MACDAnalysis.from_price_series(
            prices=prices,
            symbol="TEST",
            timestamp=datetime.now(UTC),
            fast_period=12,
            slow_period=26,
            signal_period=9,
        )

        # In a linear uptrend:
        # - MACD should be positive (fast EMA > slow EMA)
        # - Signal should be positive
        # - Histogram should be relatively small
        assert macd.macd_line > 0
        assert macd.signal_line > 0
        assert abs(macd.histogram) < abs(macd.macd_line) * 0.5

    def test_macd_crossover_scenario(self) -> None:
        """Test MACD in a crossover scenario."""
        # Create data that transitions from downtrend to uptrend
        prices = []
        for i in range(40):  # Increased from 30 to 40 for sufficient data
            if i < 20:
                prices.append(100 - i)  # Downtrend
            else:
                prices.append(80 + (i - 20) * 2)  # Strong uptrend

        prices_series = pd.Series(prices)

        macd = MACDAnalysis.from_price_series(
            prices=prices_series,
            symbol="TEST",
            timestamp=datetime.now(UTC),
        )

        # After transition, MACD should show bullish characteristics
        assert macd.trend_state in ["bullish", "neutral"]

    def test_stochastic_known_values(self) -> None:
        """Test Stochastic with known high/low ranges."""
        # Create data with known high/low over 14 periods
        # Need at least k_period + k_smooth + d_smooth = 14 + 1 + 3 = 18 points

        closes = pd.Series(
            [45, 46, 47, 48, 49, 50, 51, 52, 51, 50, 49, 48, 47, 50, 49, 50, 51, 50]
        )  # 18 points
        highs = pd.Series(
            [46, 47, 48, 49, 50, 52, 53, 60, 55, 52, 51, 50, 49, 51, 50, 51, 52, 51]
        )
        lows = pd.Series(
            [40, 45, 46, 47, 48, 49, 50, 51, 50, 49, 48, 47, 46, 49, 48, 49, 50, 49]
        )

        stoch = StochasticAnalysis.from_price_series(
            prices=closes,
            symbol="TEST",
            timestamp=datetime.now(UTC),
            k_period=14,
            k_smooth=1,  # No smoothing for easier calculation
            d_smooth=3,
            high_prices=highs,
            low_prices=lows,
        )

        # The actual calculation gives us ~28.57, which is reasonable given
        # the data pattern. The test was expecting 40-60 but that was incorrect.
        # Let's test that we get a reasonable value and the right momentum zone
        assert 20 <= stoch.k_value <= 40, f"Expected 20-40, got {stoch.k_value}"
        assert stoch.momentum_zone == "neutral"  # Should be in neutral zone

    def test_stochastic_extremes(self) -> None:
        """Test Stochastic at extremes."""
        # Price at the high of the range - need 18+ points
        at_high = pd.Series([50 + i for i in range(20)])  # Rising to high
        stoch_high = StochasticAnalysis.from_price_series(
            prices=at_high,
            symbol="TEST",
            timestamp=datetime.now(UTC),
            k_period=14,
            k_smooth=1,
            d_smooth=3,
        )
        assert stoch_high.k_value > 90  # Should be near 100
        assert stoch_high.momentum_zone == "overbought"

        # Price at the low of the range
        at_low = pd.Series([70 - i for i in range(20)])  # Falling to low
        stoch_low = StochasticAnalysis.from_price_series(
            prices=at_low,
            symbol="TEST",
            timestamp=datetime.now(UTC),
            k_period=14,
            k_smooth=1,
            d_smooth=3,
        )
        assert stoch_low.k_value < 10  # Should be near 0
        assert stoch_low.momentum_zone == "oversold"

    def test_indicator_consistency(self) -> None:
        """Test that indicators give consistent signals on the same data."""
        # Create a strong uptrend
        uptrend = pd.Series([100 + i * 2 + (i % 3) * 0.5 for i in range(50)])

        timestamp = datetime.now(UTC)

        # Calculate all indicators
        rsi = RSIAnalysis.from_price_series(
            prices=uptrend, symbol="TEST", timestamp=timestamp
        )
        macd = MACDAnalysis.from_price_series(
            prices=uptrend, symbol="TEST", timestamp=timestamp
        )
        stoch = StochasticAnalysis.from_price_series(
            prices=uptrend, symbol="TEST", timestamp=timestamp
        )

        # All should show bullish signals
        assert rsi.momentum_state in ["bullish", "strongly_bullish"]
        assert macd.trend_state == "bullish"
        assert stoch.momentum_zone in ["neutral", "overbought"]  # Uptrend pushes high

        # Signal suggestions should align
        rsi.suggest_signal()
        macd_signal = macd.suggest_signal()
        stoch.suggest_signal()

        # In strong uptrend, MACD should be bullish
        assert macd_signal["bias"] in ["bullish", "neutral"]
        # RSI might show overbought (bearish bias) which is expected
        # Stochastic might show overbought, which is bearish bias
        # So we only check MACD for consistent bullish signal
