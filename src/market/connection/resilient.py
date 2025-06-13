"""
Resilient WebSocket connection with automatic reconnection.

This module provides a wrapper around the Coinbase WebSocket client that:
- Automatically reconnects on disconnection
- Uses exponential backoff for reconnection attempts
- Re-subscribes to channels after reconnection
- Handles errors gracefully without crashing
"""

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from coinbase.websocket import WSClient
from websockets.exceptions import ConnectionClosed, WebSocketException

from src.market.model.snapshot import MarketSnapshot

if TYPE_CHECKING:
    from src.market.adapters.coinbase.stream import CoinbaseStreamHandler

logger = logging.getLogger(__name__)


class ResilientWSClient:
    """
    WebSocket client with automatic reconnection and error recovery.

    This wrapper ensures continuous market data streaming by:
    - Detecting disconnections and reconnecting automatically
    - Using exponential backoff to avoid overwhelming the server
    - Maintaining subscription state across reconnections
    - Logging all connection events for monitoring
    """

    def __init__(
        self,
        symbols: list[str],
        on_snapshot: Callable[[MarketSnapshot], None],
        api_key: str | None = None,
        api_secret: str | None = None,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        backoff_factor: float = 2.0,
    ) -> None:
        """
        Initialize resilient WebSocket client.

        Args:
            symbols: List of product IDs to stream
            on_snapshot: Callback for market snapshots
            api_key: Optional API key
            api_secret: Optional API secret
            initial_backoff: Initial reconnection delay in seconds
            max_backoff: Maximum reconnection delay in seconds
            backoff_factor: Multiplier for exponential backoff

        """
        self.symbols = symbols
        self.on_snapshot = on_snapshot
        self.api_key = api_key
        self.api_secret = api_secret

        # Reconnection settings
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_factor = backoff_factor
        self.current_backoff = initial_backoff

        # Connection state
        self.client: WSClient | None = None
        self.is_running = False
        self.reconnect_task: asyncio.Task[None] | None = None
        self._handler: CoinbaseStreamHandler | None = None

    def start(self) -> None:
        """Start the resilient WebSocket connection."""
        if self.is_running:
            logger.warning("Client already running")
            return

        self.is_running = True
        self._connect()

    def stop(self) -> None:
        """Stop the WebSocket connection and cancel reconnection."""
        self.is_running = False

        if self.reconnect_task:
            self.reconnect_task.cancel()

        if self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.error(f"Error closing client: {e}")
            finally:
                self.client = None

    def _connect(self) -> None:
        """Establish WebSocket connection and subscribe to channels."""
        try:
            logger.info("Connecting to Coinbase WebSocket...")

            # Import handler here to avoid circular imports
            from src.market.adapters.coinbase.stream import CoinbaseStreamHandler

            # Create new handler for this connection
            self._handler = CoinbaseStreamHandler(self.on_snapshot)

            # Create new client
            self.client = WSClient(
                api_key=self.api_key,
                api_secret=self.api_secret,
                on_message=self._on_message,
                on_open=self._on_open,
                on_close=self._on_close,
                on_error=self._on_error,
                retry=False,  # We handle retry logic ourselves
                verbose=False,
            )

            # Open connection
            self.client.open()

            # Subscribe to channels
            self._subscribe()

            # Reset backoff on successful connection
            self.current_backoff = self.initial_backoff
            logger.info("Successfully connected and subscribed")

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._schedule_reconnect()

    def _subscribe(self) -> None:
        """Subscribe to all configured channels."""
        if not self.client:
            return

        try:
            logger.info(f"Subscribing to channels for {self.symbols}")
            self.client.ticker(product_ids=self.symbols)
            self.client.level2(product_ids=self.symbols)
            self.client.market_trades(product_ids=self.symbols)
        except Exception as e:
            logger.error(f"Subscription error: {e}")
            self._schedule_reconnect()

    def _on_message(self, msg: str) -> None:
        """Handle incoming WebSocket messages."""
        if self._handler:
            try:
                self._handler.handle_message(msg)
            except Exception as e:
                logger.error(f"Message handling error: {e}")
                # Don't reconnect on message errors, just log

    def _on_open(self) -> None:
        """Handle WebSocket connection opened."""
        logger.info("WebSocket connection opened")

    def _on_close(self) -> None:
        """Handle WebSocket connection closed."""
        logger.warning("WebSocket connection closed")
        if self.is_running:
            self._schedule_reconnect()

    def _on_error(self, error: Exception) -> None:
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {error}")
        if isinstance(error, ConnectionClosed | WebSocketException):
            if self.is_running:
                self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        """Schedule a reconnection attempt with exponential backoff."""
        if not self.is_running:
            return

        logger.info(f"Scheduling reconnection in {self.current_backoff} seconds...")

        # Clean up existing client
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None

        # Schedule reconnection
        if self.reconnect_task:
            self.reconnect_task.cancel()

        self.reconnect_task = asyncio.create_task(self._reconnect())

    async def _reconnect(self) -> None:
        """Perform reconnection with backoff."""
        await asyncio.sleep(self.current_backoff)

        # Increase backoff for next attempt
        self.current_backoff = min(
            self.current_backoff * self.backoff_factor, self.max_backoff
        )

        # Attempt reconnection
        if self.is_running:
            logger.info("Attempting reconnection...")
            self._connect()


def stream_resilient_market_data(
    symbols: list[str],
    on_snapshot: Callable[[MarketSnapshot], None],
    api_key: str | None = None,
    api_secret: str | None = None,
) -> ResilientWSClient:
    """
    Stream market data with automatic reconnection.

    This is a drop-in replacement for stream_coinbase_market_data that adds:
    - Automatic reconnection on disconnection
    - Exponential backoff for reconnection attempts
    - Better error handling and logging

    Args:
        symbols: List of product IDs to stream
        on_snapshot: Callback for market snapshots
        api_key: Optional API key
        api_secret: Optional API secret

    Returns:
        ResilientWSClient instance for connection management

    """
    client = ResilientWSClient(
        symbols=symbols,
        on_snapshot=on_snapshot,
        api_key=api_key,
        api_secret=api_secret,
    )

    client.start()
    return client
