"""
Technical analysis service for calculating indicators.

This service manages the calculation of technical indicators with caching,
batch processing, and integration with the indicator registry.
"""

import asyncio
from collections import defaultdict
from datetime import UTC, datetime, timedelta

import pandas as pd

from src.market.analysis.base import BaseIndicator
from src.market.analysis.registry import IndicatorRegistry
from src.market.config import IndicatorConfig, ServiceConfig
from src.market.model.snapshot import MarketSnapshot
from src.market.service.models import CacheStats, IndicatorResult


class SimpleCache:
    """Simple TTL cache implementation."""

    def __init__(self, maxsize: int, ttl: int) -> None:
        """Initialize cache with max size and TTL in seconds."""
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: dict[str, tuple[BaseIndicator, datetime]] = {}

    def get(self, key: str) -> BaseIndicator | None:
        """Get item from cache if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now(UTC) - timestamp < timedelta(seconds=self.ttl):
                return value
            else:
                del self._cache[key]
        return None

    def __setitem__(self, key: str, value: BaseIndicator) -> None:
        """Add item to cache with current timestamp."""
        # Remove oldest items if at capacity
        if len(self._cache) >= self.maxsize:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        self._cache[key] = (value, datetime.now(UTC))

    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None

    def __len__(self) -> int:
        """Return number of items in cache."""
        return len(self._cache)

    def __bool__(self) -> bool:
        """Return True to indicate cache exists."""
        return True

    def clear(self) -> None:
        """Clear all items from cache."""
        self._cache.clear()


class TechnicalAnalysisService:
    """
    Service for calculating technical indicators.

    Features:
    - Caching of calculated indicators
    - Batch processing for efficiency
    - Dynamic indicator selection via configuration
    - Thread-safe operations
    """

    def __init__(
        self,
        indicator_config: IndicatorConfig,
        service_config: ServiceConfig,
        registry: IndicatorRegistry | None = None,
    ) -> None:
        """
        Initialize the technical analysis service.

        Args:
            indicator_config: Indicator-specific configuration
            service_config: Service-level configuration
            registry: Indicator registry (defaults to global registry)

        """
        self.indicator_config = indicator_config
        self.service_config = service_config
        self.registry = registry or IndicatorRegistry.default()

        # Initialize cache if enabled
        self._cache: SimpleCache | None
        if service_config.cache_enabled:
            self._cache = SimpleCache(
                maxsize=service_config.cache_max_size,
                ttl=service_config.cache_ttl_seconds,
            )
        else:
            self._cache = None

        # Price and volume series storage (symbol -> Series)
        self._price_series: dict[str, pd.Series] = defaultdict(pd.Series)
        self._volume_series: dict[str, pd.Series] = defaultdict(pd.Series)
        self._series_lock = asyncio.Lock()

    async def calculate_indicators(
        self,
        symbol: str,
        snapshot: MarketSnapshot,
    ) -> IndicatorResult:
        """
        Calculate all enabled indicators for a symbol.

        Args:
            symbol: Trading symbol
            snapshot: Current market snapshot

        Returns:
            IndicatorResult with calculated indicators and any errors

        """
        # Update price series
        await self._update_price_series(symbol, snapshot)

        # Get enabled indicators
        enabled_indicators = self._get_enabled_indicators()

        # Calculate indicators
        indicators = []
        errors = []

        for indicator_name in enabled_indicators:
            try:
                indicator = await self._calculate_indicator(
                    indicator_name,
                    symbol,
                    snapshot.timestamp,
                )
                if indicator:
                    indicators.append(indicator)
            except Exception as e:
                errors.append(f"Failed to calculate {indicator_name}: {e!s}")

        return IndicatorResult(
            symbol=symbol,
            timestamp=snapshot.timestamp,
            indicators=indicators,
            errors=errors,
        )

    async def calculate_specific_indicator(
        self,
        indicator_name: str,
        symbol: str,
        snapshot: MarketSnapshot,
    ) -> BaseIndicator | None:
        """
        Calculate a specific indicator.

        Args:
            indicator_name: Name of the indicator to calculate
            symbol: Trading symbol
            snapshot: Current market snapshot

        Returns:
            Calculated indicator or None if not available

        """
        # Update price series
        await self._update_price_series(symbol, snapshot)

        # Calculate indicator
        return await self._calculate_indicator(
            indicator_name,
            symbol,
            snapshot.timestamp,
        )

    async def batch_calculate(
        self,
        snapshots: dict[str, MarketSnapshot],
    ) -> dict[str, IndicatorResult]:
        """
        Calculate indicators for multiple symbols in batch.

        Args:
            snapshots: Map of symbol to market snapshot

        Returns:
            Map of symbol to IndicatorResult

        """
        # Process in batches for efficiency
        batch_size = self.service_config.batch_size
        symbols = list(snapshots.keys())
        results = {}

        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i : i + batch_size]

            # Calculate concurrently within batch
            tasks = [
                self.calculate_indicators(symbol, snapshots[symbol])
                for symbol in batch_symbols
            ]

            batch_results = await asyncio.gather(*tasks)

            for symbol, result in zip(batch_symbols, batch_results, strict=False):
                results[symbol] = result

        return results

    async def _update_price_series(
        self,
        symbol: str,
        snapshot: MarketSnapshot,
    ) -> None:
        """Update price and volume series with latest data."""
        if not snapshot.ticker:
            return

        async with self._series_lock:
            # Get current series
            price_series = self._price_series[symbol]
            volume_series = self._volume_series[symbol]

            # Add new price and volume
            new_price = float(snapshot.ticker.price)
            new_volume = (
                float(snapshot.ticker.volume) if snapshot.ticker.volume else 0.0
            )

            new_price_series = pd.concat(
                [price_series, pd.Series([new_price], index=[snapshot.timestamp])]
            )
            new_volume_series = pd.concat(
                [volume_series, pd.Series([new_volume], index=[snapshot.timestamp])]
            )

            # Limit series length (keep last 500 points)
            if len(new_price_series) > 500:
                new_price_series = new_price_series.iloc[-500:]
                new_volume_series = new_volume_series.iloc[-500:]

            self._price_series[symbol] = new_price_series
            self._volume_series[symbol] = new_volume_series

    async def _calculate_indicator(
        self,
        indicator_name: str,
        symbol: str,
        timestamp: datetime,
    ) -> BaseIndicator | None:
        """Calculate a single indicator with caching."""
        # Check cache first
        cache_key = f"{indicator_name}:{symbol}:{timestamp.isoformat()}"
        if self._cache and cache_key in self._cache:
            return self._cache.get(cache_key)

        # Get price and volume series
        async with self._series_lock:
            prices = self._price_series[symbol].copy()
            volumes = self._volume_series[symbol].copy()

        if len(prices) == 0:
            return None

        # Get indicator class
        if not self.registry.has(indicator_name):
            return None

        indicator_class = self.registry.get(indicator_name)

        # Check if we have enough data
        required_periods = self._get_required_periods(indicator_name)
        if len(prices) < required_periods:
            return None

        try:
            # Calculate indicator
            # We cast here because from_price_series is implementation-specific
            indicator: BaseIndicator

            if indicator_name == "rsi":
                indicator = indicator_class.from_price_series(  # type: ignore
                    prices=prices,
                    symbol=symbol,
                    timestamp=timestamp,
                    period=self.indicator_config.rsi_period,
                )
            elif indicator_name == "macd":
                indicator = indicator_class.from_price_series(  # type: ignore
                    prices=prices,
                    symbol=symbol,
                    timestamp=timestamp,
                    fast_period=self.indicator_config.macd_fast_period,
                    slow_period=self.indicator_config.macd_slow_period,
                    signal_period=self.indicator_config.macd_signal_period,
                )
            elif indicator_name == "stochastic":
                indicator = indicator_class.from_price_series(  # type: ignore
                    prices=prices,
                    symbol=symbol,
                    timestamp=timestamp,
                    k_period=self.indicator_config.stochastic_k_period,
                    k_smooth=self.indicator_config.stochastic_k_smooth,
                    d_smooth=self.indicator_config.stochastic_d_smooth,
                )
            elif indicator_name == "volume":
                # Volume analysis uses both price and volume data
                if len(volumes) < required_periods:
                    return None

                # Ensure same length
                min_len = min(len(prices), len(volumes))
                prices = prices.iloc[-min_len:]
                volumes = volumes.iloc[-min_len:]

                indicator = indicator_class.from_price_volume_series(  # type: ignore
                    prices=prices,
                    volumes=volumes,
                    symbol=symbol,
                    timestamp=timestamp,
                    period=self.indicator_config.volume_period,
                    calculate_profile=self.indicator_config.volume_profile_enabled,
                )
            else:
                # Generic creation for other indicators
                indicator = indicator_class.from_price_series(  # type: ignore
                    prices=prices,
                    symbol=symbol,
                    timestamp=timestamp,
                )

            # Cache result
            if self._cache:
                self._cache[cache_key] = indicator

            return indicator

        except Exception:
            # Log error (in production, use proper logging)
            # For now, silently fail and return None
            # This allows the service to continue with other indicators
            return None

    def _get_enabled_indicators(self) -> list[str]:
        """Get list of enabled indicator names."""
        enabled = []

        if self.indicator_config.rsi_enabled:
            enabled.append("rsi")
        if self.indicator_config.macd_enabled:
            enabled.append("macd")
        if self.indicator_config.stochastic_enabled:
            enabled.append("stochastic")
        if self.indicator_config.bb_enabled:
            enabled.append("bollinger_bands")
        if self.indicator_config.volume_enabled:
            enabled.append("volume")

        # Only return indicators that are registered
        return [name for name in enabled if self.registry.has(name)]

    def _get_required_periods(self, indicator_name: str) -> int:
        """Get minimum required periods for an indicator."""
        if indicator_name == "rsi":
            return self.indicator_config.rsi_period + 1
        elif indicator_name == "macd":
            return (
                self.indicator_config.macd_slow_period
                + self.indicator_config.macd_signal_period
            )
        elif indicator_name == "stochastic":
            return (
                self.indicator_config.stochastic_k_period
                + self.indicator_config.stochastic_d_smooth
            )
        elif indicator_name == "bollinger_bands":
            return self.indicator_config.bb_period
        elif indicator_name == "volume":
            return self.indicator_config.volume_period
        else:
            return 20  # Default minimum

    def clear_cache(self) -> None:
        """Clear the indicator cache."""
        if self._cache:
            self._cache.clear()

    def get_cache_stats(self) -> CacheStats:
        """Get cache statistics."""
        if not self._cache:
            return CacheStats(enabled=False)

        return CacheStats(
            enabled=True,
            size=len(self._cache),
            max_size=self._cache.maxsize,
            ttl=self._cache.ttl,
            hit_rate=0.0,  # Not tracked in simple implementation
        )

    async def get_price_series(self, symbol: str) -> pd.Series | None:
        """
        Get the current price series for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Price series or None if not available

        """
        async with self._series_lock:
            series = self._price_series.get(symbol)
            return series.copy() if series is not None else None
