"""
Market service configuration using Pydantic Settings.

This module provides configuration management for the market service,
allowing environment-based configuration with type validation and defaults.
"""

from decimal import Decimal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConnectionConfig(BaseSettings):
    """WebSocket connection configuration."""

    model_config = SettingsConfigDict(env_prefix="MARKET_CONNECTION_")

    # Connection settings
    ws_url: str = "wss://advanced-trade-ws.coinbase.com"
    api_key: str = Field(default="", description="Coinbase API key")
    api_secret: str = Field(default="", description="Coinbase API secret")

    # Resilience settings
    reconnect_interval: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Initial reconnect interval in seconds",
    )
    max_reconnect_interval: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Maximum reconnect interval in seconds",
    )
    backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Exponential backoff multiplier",
    )
    max_reconnect_attempts: int = Field(
        default=0,  # 0 = infinite
        ge=0,
        description="Maximum reconnection attempts (0 = infinite)",
    )


class AnalysisConfig(BaseSettings):
    """Analysis and threshold configuration."""

    model_config = SettingsConfigDict(env_prefix="MARKET_ANALYSIS_")

    # General thresholds
    max_spread_bps: Decimal = Field(
        default=Decimal("10"),
        description="Maximum acceptable spread in basis points",
    )
    min_liquidity_score: Decimal = Field(
        default=Decimal("70"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Minimum liquidity score (0-100)",
    )

    # Pipeline settings
    history_minutes: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Minutes of history for analysis pipeline",
    )
    snapshot_ttl_seconds: int = Field(
        default=300,
        ge=60,
        description="Snapshot time-to-live in seconds",
    )


class IndicatorConfig(BaseSettings):
    """Technical indicator configuration."""

    model_config = SettingsConfigDict(env_prefix="MARKET_INDICATOR_")

    # RSI settings
    rsi_enabled: bool = True
    rsi_period: int = Field(default=14, ge=2, le=100)
    rsi_overbought: float = Field(default=70.0, ge=50.0, le=100.0)
    rsi_oversold: float = Field(default=30.0, ge=0.0, le=50.0)

    # MACD settings
    macd_enabled: bool = True
    macd_fast_period: int = Field(default=12, ge=2, le=50)
    macd_slow_period: int = Field(default=26, ge=10, le=100)
    macd_signal_period: int = Field(default=9, ge=2, le=50)

    # Stochastic settings
    stochastic_enabled: bool = True
    stochastic_k_period: int = Field(default=14, ge=3, le=50)
    stochastic_k_smooth: int = Field(default=3, ge=1, le=10)
    stochastic_d_smooth: int = Field(default=3, ge=1, le=10)
    stochastic_overbought: float = Field(default=80.0, ge=50.0, le=100.0)
    stochastic_oversold: float = Field(default=20.0, ge=0.0, le=50.0)

    # Bollinger Bands settings
    bb_enabled: bool = False  # Not in Phase 1
    bb_period: int = Field(default=20, ge=5, le=100)
    bb_std_dev: float = Field(default=2.0, gt=0.0, le=5.0)

    # Volume Analysis settings
    volume_enabled: bool = True
    volume_period: int = Field(default=14, ge=5, le=50)
    volume_vwap_enabled: bool = Field(
        default=True, description="Calculate Volume Weighted Average Price"
    )
    volume_profile_enabled: bool = Field(
        default=False, description="Calculate volume profile for support/resistance"
    )


class ServiceConfig(BaseSettings):
    """Service layer configuration."""

    model_config = SettingsConfigDict(env_prefix="MARKET_SERVICE_")

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = Field(
        default=60,
        ge=1,
        description="Cache time-to-live in seconds",
    )
    cache_max_size: int = Field(
        default=1000,
        ge=100,
        description="Maximum cache entries",
    )

    # Performance
    batch_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Batch processing size",
    )
    worker_threads: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of worker threads",
    )


class MarketConfig(BaseSettings):
    """Root configuration combining all sub-configs."""

    model_config = SettingsConfigDict(env_prefix="MARKET_")

    # Sub-configurations
    connection: ConnectionConfig = Field(default_factory=ConnectionConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    indicators: IndicatorConfig = Field(default_factory=IndicatorConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)

    # Global settings
    debug: bool = False
    log_level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    )

    @classmethod
    def from_env(cls) -> "MarketConfig":
        """
        Load configuration from environment variables.

        Returns:
            Configured MarketConfig instance

        """
        return cls(
            connection=ConnectionConfig(),
            analysis=AnalysisConfig(),
            indicators=IndicatorConfig(),
            service=ServiceConfig(),
        )


# Global config instance
config = MarketConfig.from_env()
