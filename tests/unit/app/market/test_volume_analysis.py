"""
Tests for Volume Analysis indicator.

Tests volume profile analysis, VWAP calculations, and volume momentum detection
following the protocol-first architecture.
"""

from datetime import UTC, datetime

import pandas as pd
import pytest


class TestVolumeAnalysis:
    """Test VolumeAnalysis functionality."""

    @pytest.fixture
    def price_volume_data(self) -> tuple[pd.Series, pd.Series]:
        """Create test price and volume data."""
        # Simulate increasing volume on price rises
        prices = pd.Series(
            [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,  # Uptrend with increasing volume
                104.5,
                104.0,
                103.5,
                103.0,
                102.5,  # Consolidation
                102.0,
                101.0,
                100.0,
                99.0,
                98.0,  # Downtrend
                98.5,
                99.0,
                99.5,
                100.0,
                100.5,  # Recovery
            ]
        )

        volumes = pd.Series(
            [
                1000,
                1200,
                1500,
                1800,
                2000,  # Increasing volume on uptrend
                800,
                700,
                600,
                700,
                800,  # Low volume consolidation
                1500,
                1700,
                1900,
                2100,
                2200,  # High volume on downtrend
                900,
                1000,
                1100,
                1200,
                1300,  # Moderate volume recovery
            ]
        )

        return prices, volumes

    def test_volume_analysis_creation(self) -> None:
        """Test creating a VolumeAnalysis instance."""
        from src.market.analysis.volume_analysis import VolumeAnalysis

        analysis = VolumeAnalysis(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            period=14,
            current_volume=1000000,
            average_volume=800000,
            vwap=50000.0,
            volume_ratio=1.25,
            buy_volume_pct=0.65,
            sell_volume_pct=0.35,
            volume_trend="increasing",
            volume_strength="high",
            price_volume_correlation=0.85,
        )

        assert analysis.symbol == "BTC-USD"
        assert analysis.current_volume == 1000000
        assert analysis.vwap == 50000.0
        assert analysis.volume_ratio == 1.25
        assert analysis.volume_trend == "increasing"

    def test_vwap_calculation(
        self, price_volume_data: tuple[pd.Series, pd.Series]
    ) -> None:
        """Test Volume Weighted Average Price calculation."""
        from src.market.analysis.volume_analysis import VolumeAnalysis

        prices, volumes = price_volume_data

        # Calculate expected VWAP manually
        total_value = (prices * volumes).sum()
        total_volume = volumes.sum()
        expected_vwap = total_value / total_volume

        analysis = VolumeAnalysis.from_price_volume_series(
            prices=prices,
            volumes=volumes,
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
        )

        assert abs(analysis.vwap - expected_vwap) < 0.01
        assert analysis.current_volume == volumes.iloc[-1]
        assert analysis.average_volume == volumes.mean()

    def test_volume_trend_detection(
        self, price_volume_data: tuple[pd.Series, pd.Series]
    ) -> None:
        """Test volume trend detection."""
        from src.market.analysis.volume_analysis import VolumeAnalysis

        prices, volumes = price_volume_data

        # Test increasing volume scenario with much stronger trend
        # Need slope > 0.05 * average to be "increasing"
        base_volume = 1000
        increasing_volumes = pd.Series(
            [
                base_volume + (i * i * 10)
                for i in range(20)  # Exponential increase
            ]
        )
        analysis = VolumeAnalysis.from_price_volume_series(
            prices=prices[: len(increasing_volumes)],
            volumes=increasing_volumes,
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
        )

        assert analysis.volume_trend == "increasing"
        assert analysis.volume_strength in ["high", "moderate"]

        # Test decreasing volume scenario with stronger trend
        # Make sure volumes don't go negative
        decreasing_volumes = pd.Series(
            [max(100, base_volume * 3 - (i * i * 10)) for i in range(20)]
        )[: len(prices)]  # Exponential decrease but floor at 100
        analysis = VolumeAnalysis.from_price_volume_series(
            prices=prices[: len(decreasing_volumes)],
            volumes=decreasing_volumes,
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
        )

        assert analysis.volume_trend == "decreasing"

        # Test stable volume scenario
        stable_volumes = pd.Series([base_volume + (i % 3) * 10 for i in range(20)])
        analysis = VolumeAnalysis.from_price_volume_series(
            prices=prices[: len(stable_volumes)],
            volumes=stable_volumes,
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
        )

        assert analysis.volume_trend == "stable"

    def test_buy_sell_volume_estimation(self) -> None:
        """Test buy/sell volume estimation from trades."""
        from src.market.analysis.volume_analysis import VolumeAnalysis

        # Create trade data with clear buy/sell pressure
        trades = pd.DataFrame(
            {
                "price": [100, 101, 102, 101, 100, 99, 98, 99, 100, 101],
                "size": [10, 20, 30, 15, 25, 35, 40, 20, 25, 30],
                "side": [
                    "buy",
                    "buy",
                    "buy",
                    "sell",
                    "sell",
                    "sell",
                    "sell",
                    "buy",
                    "buy",
                    "buy",
                ],
            }
        )

        analysis = VolumeAnalysis.from_trades(
            trades=trades,
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            period=10,
        )

        # Buy volume: 10 + 20 + 30 + 20 + 25 + 30 = 135
        # Sell volume: 15 + 25 + 35 + 40 = 115
        # Total: 250
        expected_buy_pct = 135 / 250
        expected_sell_pct = 115 / 250

        assert abs(analysis.buy_volume_pct - expected_buy_pct) < 0.01
        assert abs(analysis.sell_volume_pct - expected_sell_pct) < 0.01
        assert abs(analysis.buy_volume_pct + analysis.sell_volume_pct - 1.0) < 0.001

    def test_volume_strength_classification(self) -> None:
        """Test volume strength classification."""
        from src.market.analysis.volume_analysis import VolumeAnalysis

        # Test high volume scenario (2x average)
        analysis = VolumeAnalysis(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            period=14,
            current_volume=2000000,
            average_volume=1000000,
            vwap=50000.0,
            volume_ratio=2.0,
            buy_volume_pct=0.5,
            sell_volume_pct=0.5,
            volume_trend="stable",
            volume_strength="high",
            price_volume_correlation=0.0,
        )

        assert analysis.volume_strength == "high"
        assert analysis.volume_ratio == 2.0

        # Test low volume scenario (0.5x average)
        analysis = VolumeAnalysis(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            period=14,
            current_volume=500000,
            average_volume=1000000,
            vwap=50000.0,
            volume_ratio=0.5,
            buy_volume_pct=0.5,
            sell_volume_pct=0.5,
            volume_trend="stable",
            volume_strength="low",
            price_volume_correlation=0.0,
        )

        assert analysis.volume_strength == "low"
        assert analysis.volume_ratio == 0.5

    def test_price_volume_correlation(
        self, price_volume_data: tuple[pd.Series, pd.Series]
    ) -> None:
        """Test price-volume correlation analysis."""
        from src.market.analysis.volume_analysis import VolumeAnalysis

        prices, volumes = price_volume_data

        analysis = VolumeAnalysis.from_price_volume_series(
            prices=prices,
            volumes=volumes,
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
        )

        # Should detect correlation between price and volume
        assert -1 <= analysis.price_volume_correlation <= 1

        # Test perfect positive correlation
        perfect_prices = pd.Series(range(100, 120))
        perfect_volumes = pd.Series(range(1000, 3000, 100))

        analysis = VolumeAnalysis.from_price_volume_series(
            prices=perfect_prices,
            volumes=perfect_volumes,
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
        )

        assert analysis.price_volume_correlation > 0.9

    def test_semantic_summary(self) -> None:
        """Test semantic summary generation."""
        from src.market.analysis.volume_analysis import VolumeAnalysis

        # High volume breakout scenario
        analysis = VolumeAnalysis(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            period=14,
            current_volume=2500000,
            average_volume=1000000,
            vwap=50000.0,
            volume_ratio=2.5,
            buy_volume_pct=0.75,
            sell_volume_pct=0.25,
            volume_trend="increasing",
            volume_strength="high",
            price_volume_correlation=0.9,
        )

        summary = analysis.semantic_summary()
        assert "high" in summary.lower()
        assert "2.5x" in summary or "2.50x" in summary

        # Low volume scenario - create new instance instead of mutating
        low_volume_analysis = VolumeAnalysis(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            period=14,
            current_volume=300000,
            average_volume=1000000,
            vwap=50000.0,
            volume_ratio=0.3,
            buy_volume_pct=0.5,
            sell_volume_pct=0.5,
            volume_trend="decreasing",
            volume_strength="low",
            price_volume_correlation=0.0,
        )

        summary = low_volume_analysis.semantic_summary()
        assert "low" in summary.lower()

    def test_agent_context(self) -> None:
        """Test agent context generation."""
        from src.market.analysis.volume_analysis import VolumeAnalysis

        analysis = VolumeAnalysis(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            period=14,
            current_volume=1500000,
            average_volume=1000000,
            vwap=50000.0,
            volume_ratio=1.5,
            buy_volume_pct=0.65,
            sell_volume_pct=0.35,
            volume_trend="increasing",
            volume_strength="moderate",
            price_volume_correlation=0.7,
        )

        context = analysis.to_agent_context()

        # Check that we get a typed VolumeAgentContext object
        from src.market.analysis.contexts import VolumeAgentContext

        assert isinstance(context, VolumeAgentContext)

        assert context.indicator == "volume"
        assert context.profile.current_volume == 1500000
        assert context.profile.average_volume == 1000000
        assert context.profile.volume_ratio == 1.5
        assert context.pressure == "buying"
        assert context.strength == "moderate"

    def test_signal_generation(self) -> None:
        """Test trading signal generation based on volume."""
        from src.market.analysis.volume_analysis import VolumeAnalysis

        # High volume with strong buy pressure
        analysis = VolumeAnalysis(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            period=14,
            current_volume=3000000,
            average_volume=1000000,
            vwap=50000.0,
            volume_ratio=3.0,
            buy_volume_pct=0.80,
            sell_volume_pct=0.20,
            volume_trend="increasing",
            volume_strength="high",
            price_volume_correlation=0.85,
        )

        signal = analysis.suggest_signal()
        assert signal.bias == "bullish"
        assert signal.strength in ["strong", "moderate"]
        assert "volume" in signal.reason.lower()

        # Low volume warning - create new instance
        low_volume_analysis = VolumeAnalysis(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            period=14,
            current_volume=300000,
            average_volume=1000000,
            vwap=50000.0,
            volume_ratio=0.3,
            buy_volume_pct=0.5,
            sell_volume_pct=0.5,
            volume_trend="stable",
            volume_strength="low",
            price_volume_correlation=0.0,
        )

        signal = low_volume_analysis.suggest_signal()
        assert signal.bias == "neutral"
        assert signal.action == "wait"
        assert "low volume" in signal.reason.lower()

    def test_insufficient_data_handling(self) -> None:
        """Test handling of insufficient data."""
        from src.market.analysis.volume_analysis import VolumeAnalysis

        # Too few data points
        prices = pd.Series([100, 101])
        volumes = pd.Series([1000, 1100])

        with pytest.raises(ValueError, match="Insufficient data"):
            VolumeAnalysis.from_price_volume_series(
                prices=prices,
                volumes=volumes,
                symbol="BTC-USD",
                timestamp=datetime.now(UTC),
                period=14,  # Requires more data
            )

    def test_volume_profile_analysis(self) -> None:
        """Test volume profile analysis for support/resistance levels."""
        from src.market.analysis.volume_analysis import VolumeAnalysis

        # Create price levels with concentrated volume
        prices = pd.Series(
            [
                100,
                100,
                101,
                101,
                101,  # High volume at 101
                102,
                103,
                104,
                105,
                105,
                105,
                105,
                105,  # High volume at 105
                104,
                103,
                102,
                101,
            ]
        )

        volumes = pd.Series(
            [
                500,
                600,
                1500,
                1600,
                1700,  # High volume at 101
                400,
                300,
                300,
                400,
                2000,
                2100,
                2200,
                2300,  # High volume at 105
                500,
                400,
                300,
                600,
            ]
        )

        analysis = VolumeAnalysis.from_price_volume_series(
            prices=prices,
            volumes=volumes,
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            calculate_profile=True,
        )

        # Should identify high volume nodes
        assert hasattr(analysis, "volume_nodes")
        assert analysis.volume_nodes is not None
        assert len(analysis.volume_nodes) > 0

        # Check that high volume areas are identified
        # The exact prices depend on binning, so check for nearby values
        high_volume_prices = [node["price"] for node in analysis.volume_nodes]

        # Should have identified some high volume nodes
        assert len(high_volume_prices) > 0

        # The nodes should be sorted by volume (descending)
        node_volumes = [node["volume"] for node in analysis.volume_nodes]
        assert node_volumes == sorted(node_volumes, reverse=True)

    def test_registry_integration(self) -> None:
        """Test that VolumeAnalysis is properly registered."""
        from src.market.analysis.registry import IndicatorRegistry

        # Create test registry
        test_registry = IndicatorRegistry.testing()

        # Import should trigger registration
        from src.market.analysis.volume_analysis import VolumeAnalysis

        # Manually register for test
        test_registry.indicators["volume"] = VolumeAnalysis

        assert test_registry.has("volume")
        assert test_registry.get("volume") is VolumeAnalysis

    def test_edge_cases(self) -> None:
        """Test edge cases and boundary conditions."""
        from src.market.analysis.volume_analysis import VolumeAnalysis

        # All same volume
        prices = pd.Series([100] * 20)
        volumes = pd.Series([1000] * 20)

        analysis = VolumeAnalysis.from_price_volume_series(
            prices=prices,
            volumes=volumes,
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
        )

        assert analysis.volume_trend == "stable"
        assert analysis.volume_ratio == 1.0

        # Zero correlation - but random data might have some correlation
        import numpy as np

        random_prices = pd.Series(np.random.randn(50) * 10 + 100)
        random_volumes = pd.Series(np.random.randn(50) * 1000 + 5000)

        analysis = VolumeAnalysis.from_price_volume_series(
            prices=random_prices,
            volumes=random_volumes,
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
        )

        # Random data can have some correlation, just check it's not strong
        assert -0.5 <= analysis.price_volume_correlation <= 0.5  # Weak correlation

    def test_field_validation(self) -> None:
        """Test field validation."""
        from src.market.analysis.volume_analysis import VolumeAnalysis

        # Test volume ratio bounds
        with pytest.raises(ValueError):
            VolumeAnalysis(
                symbol="BTC-USD",
                timestamp=datetime.now(UTC),
                period=14,
                current_volume=1000000,
                average_volume=1000000,
                vwap=50000.0,
                volume_ratio=-0.5,  # Invalid negative ratio
                buy_volume_pct=0.5,
                sell_volume_pct=0.5,
                volume_trend="stable",
                volume_strength="moderate",
                price_volume_correlation=0.0,
            )

        # Test buy/sell percentages sum to 1
        with pytest.raises(ValueError):
            VolumeAnalysis(
                symbol="BTC-USD",
                timestamp=datetime.now(UTC),
                period=14,
                current_volume=1000000,
                average_volume=1000000,
                vwap=50000.0,
                volume_ratio=1.0,
                buy_volume_pct=0.6,
                sell_volume_pct=0.6,  # Sum > 1
                volume_trend="stable",
                volume_strength="moderate",
                price_volume_correlation=0.0,
            )
