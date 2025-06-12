"""
Simple market data streaming service.

This module provides a clean, exchange-agnostic interface for streaming
market data. It delegates to exchange-specific implementations while
providing a consistent interface to the rest of the application.
"""

from collections.abc import Callable

from coinbase.websocket import WSClient

from src.market.adapters.coinbase.stream import stream_coinbase_market_data
from src.market.model.snapshot import MarketSnapshot


def stream_market_data(
    symbols: list[str],
    on_snapshot: Callable[[MarketSnapshot], None],
    exchange: str = "coinbase",
    api_key: str | None = None,
    api_secret: str | None = None,
) -> WSClient:
    """
    Stream market data from the specified exchange.

    This is the main entry point for market data streaming. It provides
    a simple interface that hides exchange-specific details.

    Args:
        symbols: List of trading pairs to stream
        on_snapshot: Callback for market snapshot updates
        exchange: Exchange to stream from (currently only "coinbase")
        api_key: Optional API key for authenticated data
        api_secret: Optional API secret

    Returns:
        WSClient for connection management (Coinbase-specific for now)

    Raises:
        ValueError: If exchange is not supported

    """
    match exchange.lower():
        case "coinbase":
            return stream_coinbase_market_data(
                symbols=symbols,
                on_snapshot=on_snapshot,
                api_key=api_key,
                api_secret=api_secret,
            )
        case _:
            raise ValueError(f"Unsupported exchange: {exchange}")
