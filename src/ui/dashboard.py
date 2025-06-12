"""Market Service Dashboard - AI-powered trading analysis with maximum data visibility."""  # noqa: E501

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar, cast

from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Footer, Header

from src.market.analysis.base import BaseIndicator
from src.market.analysis.registry import IndicatorRegistry
from src.market.config import IndicatorConfig, ServiceConfig
from src.market.model.book import OrderBook
from src.market.model.snapshot import MarketSnapshot
from src.market.model.ticker import MarketTicker
from src.market.model.trade import MarketTrade
from src.market.service.technical_analysis import TechnicalAnalysisService

if TYPE_CHECKING:
    from asyncio import Task

    from coinbase.websocket import WSClient


class OrderBookDepthWidget(Widget):
    """Enhanced order book with depth visualization and metrics."""

    order_book: reactive[OrderBook | None] = reactive(None)

    def render(self) -> RenderableType:
        """Render order book with visual depth bars and key metrics."""
        if not self.order_book:
            return "Awaiting order book data..."

        # Calculate key metrics
        best_bid = (
            self.order_book.bid_levels[0].price
            if self.order_book.bid_levels
            else Decimal("0")
        )
        best_ask = (
            self.order_book.ask_levels[0].price
            if self.order_book.ask_levels
            else Decimal("0")
        )
        spread = best_ask - best_bid
        spread_bps = (spread / best_bid * 10000) if best_bid > 0 else Decimal("0")

        # Calculate depth metrics
        bid_depth_5 = sum(level.size for level in self.order_book.bid_levels[:5])
        ask_depth_5 = sum(level.size for level in self.order_book.ask_levels[:5])
        depth_imbalance = (
            (bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5)
            if (bid_depth_5 + ask_depth_5) > 0
            else 0
        )

        # Build header with metrics
        header = Text()
        header.append(
            f"Spread: ${spread:.2f} ({spread_bps:.1f}bps) | ", style="bold yellow"
        )

        if depth_imbalance > 0.2:
            header.append(f"Bid Heavy ({depth_imbalance:.1%})", style="bold green")
        elif depth_imbalance < -0.2:
            header.append(f"Ask Heavy ({abs(depth_imbalance):.1%})", style="bold red")
        else:
            header.append("Balanced", style="bold white")

        # Create depth visualization
        max_size = max(
            max(
                (level.size for level in self.order_book.bid_levels[:6]),
                default=Decimal("0"),
            ),
            max(
                (level.size for level in self.order_book.ask_levels[:6]),
                default=Decimal("0"),
            ),
        )

        rows = []
        for i in range(6):
            bid_level = (
                self.order_book.bid_levels[i]
                if i < len(self.order_book.bid_levels)
                else None
            )
            ask_level = (
                self.order_book.ask_levels[i]
                if i < len(self.order_book.ask_levels)
                else None
            )

            # Bid side
            if bid_level:
                bid_bar_width = (
                    int((bid_level.size / max_size) * 20) if max_size > 0 else 0
                )
                bid_bar = "█" * bid_bar_width + "░" * (20 - bid_bar_width)
                bid_text = f"{bid_level.size:>7.3f} {bid_bar} ${bid_level.price:>9.2f}"
            else:
                bid_text = " " * 40

            # Ask side
            if ask_level:
                ask_bar_width = (
                    int((ask_level.size / max_size) * 20) if max_size > 0 else 0
                )
                ask_bar = "█" * ask_bar_width + "░" * (20 - ask_bar_width)
                ask_text = f"${ask_level.price:<9.2f} {ask_bar} {ask_level.size:<7.3f}"
            else:
                ask_text = " " * 40

            # Color based on distance from mid
            if i == 0:
                row = Text(f"{bid_text} │ {ask_text}")
                row.stylize("bold green", 0, 40)
                row.stylize("bold red", 43, 83)
            else:
                opacity = max(0.3, 1.0 - (i * 0.1))
                row = Text(f"{bid_text} │ {ask_text}")
                row.stylize(f"green{int(opacity * 10)}", 0, 40)
                row.stylize(f"red{int(opacity * 10)}", 43, 83)

            rows.append(row)

        # Combine everything
        content = Group(header, Text("─" * 83), *rows)

        return Panel(content, title="Order Book Depth", border_style="blue", padding=0)


class CompactIndicatorGrid(Widget):
    """Ultra-compact indicator grid showing all values at once."""

    indicators: reactive[list[BaseIndicator]] = reactive(list)

    def render(self) -> RenderableType:
        """Render all indicators in a dense grid format."""
        if not self.indicators:
            return "Calculating indicators..."

        from src.market.analysis.macd import MACDAnalysis
        from src.market.analysis.momentum_indicators import RSIAnalysis
        from src.market.analysis.stochastic import StochasticAnalysis
        from src.market.analysis.volume_analysis import VolumeAnalysis

        # Create mini panels for each indicator
        panels = []

        for indicator in self.indicators:
            signal = indicator.suggest_signal()
            bias_color = {"bullish": "green", "bearish": "red", "neutral": "yellow"}[
                signal["bias"]
            ]

            if isinstance(indicator, RSIAnalysis):
                # RSI with visual gauge
                rsi = indicator.rsi_value
                gauge = self._create_gauge(rsi, 0, 100, [30, 70])
                content = Text()
                content.append(f"RSI {rsi:.1f}\n", style=f"bold {bias_color}")
                content.append(gauge + "\n")
                content.append(f"{indicator.momentum_state}\n", style="dim")
                content.append(
                    f"Divergence: {'Yes' if indicator.divergence_detected else 'No'}",
                    style="dim",
                )
                panels.append(
                    Panel(content, title="RSI", border_style=bias_color, height=6)
                )

            elif isinstance(indicator, MACDAnalysis):
                # MACD with histogram visualization
                hist_bar = "▲" if indicator.histogram > 0 else "▼"
                hist_size = min(5, int(abs(indicator.histogram) / 2))
                content = Text()
                content.append(
                    f"MACD: {indicator.macd_line:.2f}\n", style=f"bold {bias_color}"
                )
                content.append(f"Signal: {indicator.signal_line:.2f}\n", style="dim")
                content.append(
                    f"Hist: {hist_bar * hist_size} {indicator.histogram:.2f}\n",
                    style="green" if indicator.histogram > 0 else "red",
                )
                if indicator.crossover_detected:
                    content.append(
                        f"⚡ {indicator.crossover_type.upper()} CROSS",
                        style=f"bold {bias_color}",
                    )
                panels.append(
                    Panel(content, title="MACD", border_style=bias_color, height=6)
                )

            elif isinstance(indicator, StochasticAnalysis):
                # Stochastic with K/D relationship
                k, d = indicator.k_value, indicator.d_value
                content = Text()
                content.append(f"%K: {k:.1f} %D: {d:.1f}\n", style=f"bold {bias_color}")
                content.append(self._create_dual_gauge(k, d) + "\n")
                content.append(f"{indicator.momentum_zone}\n", style="dim")
                if indicator.crossover_detected:
                    content.append(
                        f"⚡ {indicator.crossover_type.upper()}",
                        style=f"bold {bias_color}",
                    )
                panels.append(
                    Panel(
                        content, title="Stochastic", border_style=bias_color, height=6
                    )
                )

            elif isinstance(indicator, VolumeAnalysis):
                # Volume with profile
                content = Text()
                content.append(
                    f"Vol: {indicator.volume_ratio:.2f}x avg\n",
                    style=f"bold {bias_color}",
                )
                content.append(f"VWAP: ${indicator.vwap:,.0f}\n", style="dim")
                content.append(
                    f"Buy: {indicator.buy_volume_pct:.0%} "
                    f"Sell: {indicator.sell_volume_pct:.0%}\n",
                    style="green"
                    if indicator.buy_volume_pct > 0.6
                    else "red"
                    if indicator.sell_volume_pct > 0.6
                    else "yellow",
                )
                content.append(f"Trend: {indicator.volume_trend}", style="dim")
                panels.append(
                    Panel(content, title="Volume", border_style=bias_color, height=6)
                )

        # Arrange in columns
        return Columns(panels, equal=True, expand=True)

    def _create_gauge(
        self, value: float, min_val: float, max_val: float, zones: list[float]
    ) -> str:
        """Create a visual gauge for a value."""
        normalized = (value - min_val) / (max_val - min_val)
        position = int(normalized * 20)

        gauge = ""
        for i in range(20):
            if i == position:
                gauge += "●"
            elif i < len(zones) * 10 / len(zones) and value < zones[0]:
                gauge += "═"  # Oversold zone
            elif i > 20 - (len(zones) * 10 / len(zones)) and value > zones[-1]:
                gauge += "═"  # Overbought zone
            else:
                gauge += "─"

        return f"[{gauge}]"

    def _create_dual_gauge(self, value1: float, value2: float) -> str:
        """Create a dual gauge showing two values."""
        pos1 = int((value1 / 100) * 20)
        pos2 = int((value2 / 100) * 20)

        gauge = ""
        for i in range(20):
            if i == pos1 and i == pos2:
                gauge += "◆"
            elif i == pos1:
                gauge += "●"
            elif i == pos2:
                gauge += "○"
            else:
                gauge += "─"

        return f"[{gauge}]"


class MarketProfileWidget(Widget):
    """Market profile showing price levels, volume nodes, and key levels."""

    order_book: reactive[OrderBook | None] = reactive(None)
    indicators: reactive[list[BaseIndicator]] = reactive(list)
    ticker: reactive[MarketTicker | None] = reactive(None)

    def render(self) -> RenderableType:
        """Render market profile with key levels."""
        if not self.ticker:
            return "Awaiting market data..."

        current_price = float(self.ticker.price)

        # Extract key levels from indicators
        levels = []

        # Add VWAP if available
        from src.market.analysis.volume_analysis import VolumeAnalysis

        for ind in self.indicators:
            if isinstance(ind, VolumeAnalysis):
                levels.append(("VWAP", float(ind.vwap), "cyan"))
                if ind.volume_nodes:
                    for node in ind.volume_nodes[:3]:
                        levels.append(("Vol Node", node["price"], "magenta"))

        # Add order book levels
        if (
            self.order_book
            and self.order_book.bid_levels
            and self.order_book.ask_levels
        ):
            # Find large orders (2x average size)
            bid_sizes = [level.size for level in self.order_book.bid_levels[:20]]
            ask_sizes = [level.size for level in self.order_book.ask_levels[:20]]
            avg_size = (sum(bid_sizes) + sum(ask_sizes)) / (
                len(bid_sizes) + len(ask_sizes)
            )

            for level in self.order_book.bid_levels[:10]:
                if level.size > avg_size * 2:
                    levels.append(("Big Bid", float(level.price), "green"))
                    break

            for level in self.order_book.ask_levels[:10]:
                if level.size > avg_size * 2:
                    levels.append(("Big Ask", float(level.price), "red"))
                    break

        # Sort levels by distance from current price
        levels.sort(key=lambda x: abs(x[1] - current_price))

        # Create price ladder
        price_range = current_price * 0.002  # 0.2% range
        current_price - price_range
        max_price = current_price + price_range

        rows = []
        for i in range(12):
            price = max_price - (i * (price_range * 2) / 12)

            # Check if any key level is near this price
            level_text = ""
            level_color = "white"
            for name, level_price, color in levels:
                if abs(price - level_price) < price_range / 24:
                    level_text = f" ← {name}"
                    level_color = color
                    break

            # Create price row
            if abs(price - current_price) < price_range / 48:
                row = Text(f"  ▶ ${price:,.2f} ◀  {level_text}", style="bold yellow")
            else:
                distance_pct = ((price - current_price) / current_price) * 100
                row = Text(
                    f"    ${price:,.2f} ({distance_pct:+.2f}%) {level_text}",
                    style=level_color,
                )

            rows.append(row)

        return Panel(Group(*rows), title="Price Levels", border_style="cyan", padding=0)


class TradeMomentumWidget(Widget):
    """Trade flow momentum with buy/sell pressure visualization."""

    trades: reactive[list[MarketTrade]] = reactive(list)

    def render(self) -> RenderableType:
        """Render trade momentum analysis."""
        if not self.trades:
            return "No trade data..."

        # Analyze recent trades in time buckets
        now = datetime.now(UTC)
        buckets = [
            ("1m", 60),
            ("5m", 300),
            ("15m", 900),
        ]

        rows = []

        for label, seconds in buckets:
            recent_trades = [
                t for t in self.trades if (now - t.timestamp).total_seconds() < seconds
            ]
            if not recent_trades:
                continue

            buy_volume = sum(t.size for t in recent_trades if t.side.value == "buy")
            sell_volume = sum(t.size for t in recent_trades if t.side.value == "sell")
            total_volume = buy_volume + sell_volume

            if total_volume > 0:
                buy_pct = buy_volume / total_volume

                # Create pressure bar
                bar_width = 20
                buy_width = int(buy_pct * bar_width)
                sell_width = bar_width - buy_width

                bar = Text()
                bar.append("█" * buy_width, style="green")
                bar.append("█" * sell_width, style="red")

                # Stats
                avg_buy_price = (
                    sum(
                        t.price * t.size for t in recent_trades if t.side.value == "buy"
                    )
                    / buy_volume
                    if buy_volume > 0
                    else Decimal("0")
                )
                avg_sell_price = (
                    sum(
                        t.price * t.size
                        for t in recent_trades
                        if t.side.value == "sell"
                    )
                    / sell_volume
                    if sell_volume > 0
                    else Decimal("0")
                )

                row = Text()
                row.append(f"{label:>3}: ", style="bold")
                row.append(bar)
                row.append(f" B:{buy_pct:.0%} S:{(1 - buy_pct):.0%}", style="dim")
                if avg_buy_price > avg_sell_price:
                    row.append(" ↑", style="green")
                elif avg_sell_price > avg_buy_price:
                    row.append(" ↓", style="red")

                rows.append(row)

        # Add large trade alerts
        large_trades = sorted(self.trades, key=lambda t: t.size, reverse=True)[:3]
        if large_trades:
            rows.append(Text())
            rows.append(Text("Large Trades:", style="bold"))
            for trade in large_trades:
                color = "green" if trade.side.value == "buy" else "red"
                rows.append(
                    Text(f"  {trade.size:.3f} @ ${trade.price:,.2f}", style=color)
                )

        return Panel(
            Group(*rows), title="Trade Momentum", border_style="blue", padding=0
        )


class DenseMarketDashboard(App[None]):
    """Market Service - AI-powered trading analysis dashboard."""

    CSS = """
    Screen {
        background: $surface;
    }
    
    #main-content {
        height: auto;
        width: 100%;
    }
    
    #orderbook {
        width: 40%;
        height: auto;
        max-height: 20;
    }
    
    #trades {
        width: 30%;
        height: auto;
        max-height: 20;
    }
    
    #profile {
        width: 30%;
        height: auto;
        max-height: 20;
    }
    
    #indicators {
        height: 8;
        width: 100%;
    }
    
    CompactIndicatorGrid {
        margin: 0 1;
    }
    """

    # Class variable for key bindings
    BINDINGS: ClassVar[list[BindingType]] = cast(
        list[BindingType],
        [
            ("q", "quit", "Quit"),
            ("s", "toggle_symbol", "Switch Symbol"),
            ("d", "toggle_density", "Toggle Density"),
        ],
    )

    current_symbol: reactive[str] = reactive("BTC-USD")
    symbols: ClassVar[list[str]] = ["BTC-USD", "ETH-USD", "SOL-USD"]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the dashboard."""
        super().__init__(**kwargs)

        # Initialize services
        self.indicator_config = IndicatorConfig(
            rsi_enabled=True,
            macd_enabled=True,
            stochastic_enabled=True,
            volume_enabled=True,
        )

        self.service_config = ServiceConfig(
            cache_enabled=True,
            cache_ttl_seconds=2,  # Faster updates
        )

        # Create registry and register indicators
        self.registry = IndicatorRegistry.testing()
        self._register_indicators()

        self.analysis_service = TechnicalAnalysisService(
            indicator_config=self.indicator_config,
            service_config=self.service_config,
            registry=self.registry,
        )

        self.ws_client: WSClient | None = None
        self.update_task: Task[None] | None = None
        self.mock_data_task: Task[None] | None = None

    def _register_indicators(self) -> None:
        """Register all indicators with the registry."""
        from src.market.analysis.macd import MACDAnalysis
        from src.market.analysis.momentum_indicators import RSIAnalysis
        from src.market.analysis.stochastic import StochasticAnalysis
        from src.market.analysis.volume_analysis import VolumeAnalysis

        self.registry.indicators["rsi"] = RSIAnalysis
        self.registry.indicators["macd"] = MACDAnalysis
        self.registry.indicators["stochastic"] = StochasticAnalysis
        self.registry.indicators["volume"] = VolumeAnalysis

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()

        # Main content in a single horizontal layout
        with Horizontal(id="main-content"):
            yield OrderBookDepthWidget(id="orderbook")
            yield TradeMomentumWidget(id="trades")
            yield MarketProfileWidget(id="profile")

        # Indicators at the bottom
        yield CompactIndicatorGrid(id="indicators")

        yield Footer()

    async def on_mount(self) -> None:
        """Start market data streaming when app mounts."""
        self.title = f"Market Service - {self.current_symbol}"
        await self.start_market_stream()
        self.update_task = asyncio.create_task(self.update_indicators())

    async def start_market_stream(self) -> None:
        """Start streaming market data."""
        self.mock_data_task = asyncio.create_task(self.generate_mock_data())

    async def generate_mock_data(self) -> None:
        """Generate mock market data for testing."""
        import random

        from src.market.enums import TradeSide
        from src.market.model.types import PriceLevelData

        base_price: float = {"BTC-USD": 45000, "ETH-USD": 2500, "SOL-USD": 100}[
            self.current_symbol
        ]

        while True:
            # Generate more realistic price movement
            price_change = random.gauss(0, 0.0002)  # Smaller, more realistic moves
            current_price = base_price * (1 + price_change)

            # Create ticker
            ticker = MarketTicker(
                symbol=self.current_symbol,
                price=Decimal(str(current_price)),
                bid=Decimal(str(current_price - 0.50)),
                ask=Decimal(str(current_price + 0.50)),
                volume=Decimal(str(random.randint(1000, 5000))),
                timestamp=datetime.now(UTC),
            )

            # Create realistic order book with clustering
            bids = []
            asks = []

            # Add clustered liquidity
            for i in range(20):
                # Create clusters at round numbers
                bid_price = current_price - (i * 0.5)
                ask_price = current_price + (i * 0.5)

                # Larger sizes at round numbers
                size_multiplier = 3.0 if bid_price % 10 == 0 else 1.0

                bids.append(
                    PriceLevelData(
                        price=Decimal(str(bid_price)),
                        size=Decimal(str(random.uniform(0.1, 2.0) * size_multiplier)),
                    )
                )

                asks.append(
                    PriceLevelData(
                        price=Decimal(str(ask_price)),
                        size=Decimal(str(random.uniform(0.1, 2.0) * size_multiplier)),
                    )
                )

            order_book = OrderBook(
                symbol=self.current_symbol,
                timestamp=datetime.now(UTC),
                bid_levels=bids,
                ask_levels=asks,
            )

            # Create trades with momentum
            trades = []
            momentum = random.choice([-1, 0, 1])  # Bear, neutral, bull

            for _ in range(10):
                side = (
                    TradeSide.BUY
                    if random.random() < (0.5 + momentum * 0.2)
                    else TradeSide.SELL
                )
                size = random.uniform(0.01, 0.5)
                if random.random() < 0.1:  # 10% chance of large trade
                    size *= 5

                trades.append(
                    MarketTrade(
                        symbol=self.current_symbol,
                        price=Decimal(str(current_price + random.uniform(-1, 1))),
                        size=Decimal(str(size)),
                        side=side,
                        timestamp=datetime.now(UTC),
                        trade_id=str(random.randint(1000000, 9999999)),
                    )
                )

            # Create snapshot
            snapshot = MarketSnapshot(
                symbol=self.current_symbol,
                timestamp=datetime.now(UTC),
                ticker=ticker,
                order_book=order_book,
                trades=trades,
            )

            await self.handle_market_snapshot(snapshot)

            # Update base price
            base_price = current_price

            await asyncio.sleep(0.5)  # Faster updates

    async def handle_market_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Handle incoming market snapshot."""
        if snapshot.symbol != self.current_symbol:
            return

        # Update header subtitle with ticker info
        if snapshot.ticker and isinstance(snapshot.ticker, MarketTicker):
            spread = (
                snapshot.ticker.ask - snapshot.ticker.bid
                if snapshot.ticker.bid and snapshot.ticker.ask
                else Decimal("0")
            )
            spread_bps = (
                (spread / snapshot.ticker.price * 10000)
                if snapshot.ticker.price > 0
                else Decimal("0")
            )

            self.sub_title = (
                f"{snapshot.ticker.symbol} "
                f"${snapshot.ticker.price:,.2f} | "
                f"Spread: {spread_bps:.1f}bps | "
                f"Vol: {snapshot.ticker.volume:,.0f}"
            )

        # Update widgets
        if snapshot.order_book and isinstance(snapshot.order_book, OrderBook):
            self.query_one(
                "#orderbook", OrderBookDepthWidget
            ).order_book = snapshot.order_book
            self.query_one(
                "#profile", MarketProfileWidget
            ).order_book = snapshot.order_book

        if snapshot.ticker and isinstance(snapshot.ticker, MarketTicker):
            self.query_one("#profile", MarketProfileWidget).ticker = snapshot.ticker

        if snapshot.trades:
            trades_widget = self.query_one("#trades", TradeMomentumWidget)
            current_trades = list(trades_widget.trades)
            new_trades = [t for t in snapshot.trades if isinstance(t, MarketTrade)]
            trades_widget.trades = new_trades + current_trades[:100]

        # Update price series
        await self.analysis_service._update_price_series(snapshot.symbol, snapshot)

    async def update_indicators(self) -> None:
        """Update technical indicators."""
        while True:
            try:
                ticker = self.query_one("#profile", MarketProfileWidget).ticker
                if ticker:
                    snapshot = MarketSnapshot(
                        symbol=self.current_symbol,
                        timestamp=datetime.now(UTC),
                        ticker=ticker,
                    )

                    result = await self.analysis_service.calculate_indicators(
                        self.current_symbol, snapshot
                    )

                    if result.success and result.indicators:
                        self.query_one(
                            "#indicators", CompactIndicatorGrid
                        ).indicators = result.indicators
                        self.query_one(
                            "#profile", MarketProfileWidget
                        ).indicators = result.indicators

            except Exception as e:
                self.notify(f"Indicator error: {e}", severity="warning")

            await asyncio.sleep(2)  # Faster updates

    def action_toggle_symbol(self) -> None:
        """Switch to next symbol."""
        current_idx = self.symbols.index(self.current_symbol)
        next_idx = (current_idx + 1) % len(self.symbols)
        self.current_symbol = self.symbols[next_idx]

        self.title = f"Market Service - {self.current_symbol}"
        self.notify(f"Switched to {self.current_symbol}")


def run_dashboard() -> None:
    """Run the information-dense market dashboard."""
    app = DenseMarketDashboard()
    app.run()


if __name__ == "__main__":
    run_dashboard()
