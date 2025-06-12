"""
Tests for TechnicalAnalysisService.

Tests the service layer that manages indicator calculations,
caching, and batch processing.
"""

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

# Import indicators for use in tests
from src.market.analysis.macd import MACDAnalysis
from src.market.analysis.momentum_indicators import RSIAnalysis
from src.market.analysis.registry import IndicatorRegistry
from src.market.config import IndicatorConfig, ServiceConfig
from src.market.service.models import CacheStats, IndicatorResult
from src.market.service.technical_analysis import TechnicalAnalysisService


def create_mock_snapshot(
    symbol: str, price: float, timestamp: datetime | None = None
) -> Any:
    """Create a properly mocked MarketSnapshot."""
    if timestamp is None:
        timestamp = datetime.now(UTC)

    # Create snapshot mock
    snapshot = MagicMock()
    snapshot.symbol = symbol
    snapshot.timestamp = timestamp

    # Create ticker mock with price as a property
    ticker_mock = MagicMock()
    # Use PropertyMock to properly mock the price property
    from unittest.mock import PropertyMock

    type(ticker_mock).price = PropertyMock(return_value=price)
    snapshot.ticker = ticker_mock

    return snapshot


class TestTechnicalAnalysisService:
    """Test TechnicalAnalysisService functionality."""

    @pytest.fixture
    def test_registry(self) -> IndicatorRegistry:
        """Create isolated test registry with indicators."""
        registry = IndicatorRegistry.testing()

        # Manually register the indicators we need for tests
        registry.indicators["rsi"] = RSIAnalysis
        registry.indicators["macd"] = MACDAnalysis

        return registry

    @pytest.fixture
    def indicator_config(self) -> IndicatorConfig:
        """Create test indicator configuration."""
        return IndicatorConfig(
            rsi_enabled=True,
            rsi_period=14,
            macd_enabled=True,
            macd_fast_period=12,
            macd_slow_period=26,
            macd_signal_period=9,
            stochastic_enabled=False,  # Not implemented yet
            bb_enabled=False,  # Not implemented yet
        )

    @pytest.fixture
    def service_config(self) -> ServiceConfig:
        """Create test service configuration."""
        return ServiceConfig(
            cache_enabled=True,
            cache_ttl_seconds=60,
            cache_max_size=100,
            batch_size=10,
            worker_threads=2,
        )

    @pytest.fixture
    def service(
        self,
        indicator_config: IndicatorConfig,
        service_config: ServiceConfig,
        test_registry: IndicatorRegistry,
    ) -> TechnicalAnalysisService:
        """Create test service instance with isolated registry."""
        return TechnicalAnalysisService(
            indicator_config=indicator_config,
            service_config=service_config,
            registry=test_registry,
        )

    @pytest.fixture
    def mock_snapshot(self) -> Any:
        """Create mock market snapshot."""
        return create_mock_snapshot("BTC-USD", 50000.0)

    @pytest.mark.asyncio
    async def test_service_initialization(
        self,
        indicator_config: IndicatorConfig,
        service_config: ServiceConfig,
    ) -> None:
        """Test service initialization."""
        service = TechnicalAnalysisService(
            indicator_config=indicator_config,
            service_config=service_config,
        )

        assert service.indicator_config == indicator_config
        assert service.service_config == service_config
        assert service._cache is not None  # Cache enabled
        assert len(service._price_series) == 0

    @pytest.mark.asyncio
    async def test_update_price_series(
        self,
        service: TechnicalAnalysisService,
        mock_snapshot: Any,
    ) -> None:
        """Test updating price series."""
        # Update with first price
        await service._update_price_series("BTC-USD", mock_snapshot)

        # Check series was created
        series = await service.get_price_series("BTC-USD")
        assert series is not None
        assert len(series) == 1
        assert series.iloc[0] == 50000.0

        # Update with more prices
        for i in range(5):
            new_snapshot = create_mock_snapshot("BTC-USD", 50000.0 + i * 100)
            await service._update_price_series("BTC-USD", new_snapshot)

        # Check series was updated
        series = await service.get_price_series("BTC-USD")
        assert series is not None
        assert len(series) == 6

    @pytest.mark.asyncio
    async def test_calculate_indicators_insufficient_data(
        self,
        service: TechnicalAnalysisService,
        mock_snapshot: Any,
    ) -> None:
        """Test calculating indicators with insufficient data."""
        # Only one price point
        await service._update_price_series("BTC-USD", mock_snapshot)

        result = await service.calculate_indicators("BTC-USD", mock_snapshot)

        # Should return result with empty indicators list
        assert isinstance(result, IndicatorResult)
        assert result.symbol == "BTC-USD"
        assert len(result.indicators) == 0
        assert result.success is False  # No indicators calculated

    @pytest.mark.asyncio
    async def test_calculate_indicators_sufficient_data(
        self,
        service: TechnicalAnalysisService,
        mock_snapshot: Any,
    ) -> None:
        """Test calculating indicators with sufficient data."""
        # Add enough price data
        for i in range(50):
            new_snapshot = create_mock_snapshot("BTC-USD", 50000.0 + i * 10)
            await service._update_price_series("BTC-USD", new_snapshot)

        result = await service.calculate_indicators("BTC-USD", mock_snapshot)

        # Should calculate enabled indicators
        assert isinstance(result, IndicatorResult)
        assert len(result.indicators) > 0
        assert result.success is True

        # Check for RSI and MACD (both enabled)
        indicator_names = result.indicator_names
        assert "RSIAnalysis" in indicator_names
        assert "MACDAnalysis" in indicator_names

    @pytest.mark.asyncio
    async def test_calculate_specific_indicator(
        self,
        service: TechnicalAnalysisService,
        mock_snapshot: Any,
    ) -> None:
        """Test calculating a specific indicator."""
        # Add price data
        for i in range(30):
            new_snapshot = create_mock_snapshot("BTC-USD", 50000.0 + i * 10)
            await service._update_price_series("BTC-USD", new_snapshot)

        # Calculate specific indicator
        rsi = await service.calculate_specific_indicator(
            "rsi", "BTC-USD", mock_snapshot
        )

        assert rsi is not None
        assert rsi.symbol == "BTC-USD"

    @pytest.mark.asyncio
    async def test_batch_calculate(
        self,
        service: TechnicalAnalysisService,
    ) -> None:
        """Test batch calculation for multiple symbols."""
        # Create snapshots for multiple symbols
        snapshots: dict[str, Any] = {}
        for symbol in ["BTC-USD", "ETH-USD", "SOL-USD"]:
            snapshot = MagicMock(spec=Any)
            snapshot.symbol = symbol
            snapshot.timestamp = datetime.now(UTC)
            ticker_mock = MagicMock()
            ticker_mock.price = 100.0
            snapshot.ticker = ticker_mock
            snapshots[symbol] = snapshot

            # Add some price history
            for i in range(30):
                ticker_mock.price = 100.0 + i
                await service._update_price_series(symbol, snapshot)

        # Batch calculate
        results = await service.batch_calculate(snapshots)

        assert len(results) == 3
        assert all(symbol in results for symbol in ["BTC-USD", "ETH-USD", "SOL-USD"])

        # Check each result has indicators
        for symbol, result in results.items():
            assert isinstance(result, IndicatorResult)
            assert result.symbol == symbol
            assert len(result.indicators) > 0
            assert result.success is True

    @pytest.mark.asyncio
    async def test_caching(
        self,
        service: TechnicalAnalysisService,
        mock_snapshot: Any,
    ) -> None:
        """Test indicator caching."""
        # Add price data
        for i in range(30):
            new_snapshot = create_mock_snapshot("BTC-USD", 50000.0 + i * 10)
            await service._update_price_series("BTC-USD", new_snapshot)

        # Calculate indicator (should cache)
        timestamp = datetime.now(UTC)
        mock_snapshot.timestamp = timestamp

        # Clear any existing cache entries
        service.clear_cache()

        # First calculation - should compute and cache
        indicator1 = await service._calculate_indicator("rsi", "BTC-USD", timestamp)
        assert indicator1 is not None

        # Second calculation - should hit cache
        indicator2 = await service._calculate_indicator("rsi", "BTC-USD", timestamp)

        # Both should be the same cached object
        assert indicator1 is indicator2

    @pytest.mark.asyncio
    async def test_cache_disabled(
        self,
        indicator_config: IndicatorConfig,
    ) -> None:
        """Test service with cache disabled."""
        service_config = ServiceConfig(cache_enabled=False)
        service = TechnicalAnalysisService(
            indicator_config=indicator_config,
            service_config=service_config,
        )

        assert service._cache is None

        # Cache stats should show disabled
        stats = service.get_cache_stats()
        assert isinstance(stats, CacheStats)
        assert stats.enabled is False
        assert stats.size == 0
        assert stats.max_size == 0

    def test_get_enabled_indicators(
        self,
        service: TechnicalAnalysisService,
    ) -> None:
        """Test getting list of enabled indicators."""
        # Check that only registered indicators are returned
        list(service.registry.indicators.keys())

        enabled = service._get_enabled_indicators()

        # All enabled indicators should be registered
        assert "rsi" in enabled
        assert "macd" in enabled

        # Stochastic and BB are disabled
        assert "stochastic" not in enabled
        assert "bollinger_bands" not in enabled

    def test_get_required_periods(
        self,
        service: TechnicalAnalysisService,
    ) -> None:
        """Test getting required periods for indicators."""
        assert service._get_required_periods("rsi") == 15  # 14 + 1
        assert service._get_required_periods("macd") == 35  # 26 + 9
        assert service._get_required_periods("unknown") == 20  # Default

    def test_clear_cache(
        self,
        service: TechnicalAnalysisService,
    ) -> None:
        """Test clearing the cache."""
        # Skip test if cache is disabled
        if service._cache is None:
            return

        # Add something to cache using a mock indicator

        mock_indicator = RSIAnalysis(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            rsi_value=50.0,
            average_gain=0.5,
            average_loss=0.5,
            momentum_state="neutral",
            momentum_strength="moderate",
        )
        service._cache["test_key"] = mock_indicator
        assert len(service._cache) == 1

        # Clear cache
        service.clear_cache()
        assert len(service._cache) == 0

    def test_get_cache_stats(
        self,
        service: TechnicalAnalysisService,
    ) -> None:
        """Test getting cache statistics."""
        # Debug: Check if cache is properly initialized
        assert service._cache is not None, "Cache should be initialized"
        assert service.service_config.cache_enabled is True, (
            "Cache should be enabled in config"
        )

        stats = service.get_cache_stats()

        assert isinstance(stats, CacheStats)
        assert stats.enabled is True
        assert stats.size == 0  # Initially empty
        assert stats.max_size == 100
        assert stats.ttl == 60

    @pytest.mark.asyncio
    async def test_price_series_limit(
        self,
        service: TechnicalAnalysisService,
        mock_snapshot: Any,
    ) -> None:
        """Test that price series is limited to prevent memory issues."""
        # Add more than 500 prices
        for i in range(600):
            mock_snapshot.ticker.price = 50000.0 + i
            mock_snapshot.timestamp = datetime.now(UTC)
            await service._update_price_series("BTC-USD", mock_snapshot)

        # Check series is limited
        series = await service.get_price_series("BTC-USD")
        assert series is not None
        assert len(series) == 500  # Limited to last 500

    @pytest.mark.asyncio
    async def test_error_handling(
        self,
        service: TechnicalAnalysisService,
        mock_snapshot: Any,
        test_registry: IndicatorRegistry,
    ) -> None:
        """Test error handling in indicator calculation."""
        # Add price data
        for i in range(30):
            mock_snapshot.ticker.price = 50000.0 + i * 10
            mock_snapshot.timestamp = datetime.now(UTC)
            await service._update_price_series("BTC-USD", mock_snapshot)

        # Create a mock indicator that raises an error
        class ErrorIndicator:
            @staticmethod
            def from_price_series(*args: Any, **kwargs: Any) -> None:
                raise Exception("Test error")

        # Register the error-prone indicator
        test_registry.indicators["error_indicator"] = ErrorIndicator

        # Should return None instead of raising
        result = await service._calculate_indicator(
            "error_indicator", "BTC-USD", datetime.now(UTC)
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_concurrent_updates(
        self,
        service: TechnicalAnalysisService,
    ) -> None:
        """Test concurrent price series updates."""

        # Create multiple tasks updating the same symbol
        async def update_task(symbol: str, price: float) -> None:
            snapshot = MagicMock(spec=Any)
            snapshot.symbol = symbol
            snapshot.timestamp = datetime.now(UTC)
            snapshot.ticker = MagicMock()
            snapshot.ticker.price = price
            await service._update_price_series(symbol, snapshot)

        # Run concurrent updates
        tasks = [update_task("BTC-USD", 50000.0 + i) for i in range(20)]
        await asyncio.gather(*tasks)

        # Check series integrity
        series = await service.get_price_series("BTC-USD")
        assert series is not None
        assert len(series) == 20
