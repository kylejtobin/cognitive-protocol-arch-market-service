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
from src.market.domain.primitives import Price
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
                signal.bias
            ]

            if isinstance(indicator, RSIAnalysis):
                # RSI with visual gauge
                rsi = float(indicator.rsi_value)
                gauge = self._create_gauge(rsi, 0, 100, [30, 70])
                content = Text()
                content.append(f"RSI {rsi:.1f}\n", style=f"bold {bias_color}")
                content.append(gauge + "\n")
                # Format momentum state to be more readable
                momentum_display = indicator.momentum_state.replace("_", " ").title()
                content.append(f"{momentum_display}\n", style="dim")
                content.append(
                    f"Divergence: {'Yes' if indicator.divergence_detected else 'No'}",
                    style="dim",
                )
                panels.append(
                    Panel(
                        content, title="RSI", border_style=bias_color.lower(), height=6
                    )
                )

            elif isinstance(indicator, MACDAnalysis):
                # MACD with histogram visualization
                hist_bar = "▲" if indicator.histogram_value > 0 else "▼"
                hist_size = min(5, int(abs(indicator.histogram_value) / 2))
                content = Text()
                content.append(
                    f"MACD: {indicator.macd_value:.2f}\n", style=f"bold {bias_color}"
                )
                content.append(f"Signal: {indicator.signal_value:.2f}\n", style="dim")
                content.append(
                    f"Hist: {hist_bar * hist_size} {indicator.histogram_value:.2f}\n",
                    style="green" if indicator.histogram_value > 0 else "red",
                )
                if indicator.crossover_detected:
                    content.append(
                        f"⚡ {indicator.crossover_type.upper()} CROSS",
                        style=f"bold {bias_color}",
                    )
                panels.append(
                    Panel(
                        content, title="MACD", border_style=bias_color.lower(), height=6
                    )
                )

            elif isinstance(indicator, StochasticAnalysis):
                # Stochastic with K/D relationship
                k, d = indicator.k_value, indicator.d_value
                content = Text()
                content.append(f"%K: {k:.1f} %D: {d:.1f}\n", style=f"bold {bias_color}")
                content.append(self._create_dual_gauge(k, d) + "\n")
                # Format momentum zone to be more readable
                zone_display = indicator.momentum_zone.replace("_", " ").title()
                content.append(f"{zone_display}\n", style="dim")
                if indicator.crossover_detected:
                    content.append(
                        f"⚡ {indicator.crossover_type.upper()}",
                        style=f"bold {bias_color}",
                    )
                panels.append(
                    Panel(
                        content,
                        title="Stochastic",
                        border_style=bias_color.lower(),
                        height=6,
                    )
                )

            elif isinstance(indicator, VolumeAnalysis):
                # Volume with profile
                content = Text()
                content.append(
                    f"Vol: {indicator.volume_ratio:.2f}x avg\n",
                    style=f"bold {bias_color}",
                )
                # Handle NaN VWAP gracefully
                try:
                    vwap_value = float(indicator.vwap)
                    if not (vwap_value != vwap_value):  # Check for NaN
                        content.append(f"VWAP: ${vwap_value:,.0f}\n", style="dim")
                    else:
                        content.append("VWAP: N/A\n", style="dim")
                except (ValueError, TypeError, AttributeError):
                    content.append("VWAP: N/A\n", style="dim")
                content.append(
                    f"Buy: {indicator.buy_volume_pct:.0%} "
                    f"Sell: {indicator.sell_volume_pct:.0%}\n",
                    style="green"
                    if indicator.buy_volume_pct > 0.6
                    else "red"
                    if indicator.sell_volume_pct > 0.6
                    else "yellow",
                )
                # Format volume trend to be more readable
                trend_display = indicator.volume_trend.replace("_", " ").title()
                content.append(f"Trend: {trend_display}", style="dim")
                panels.append(
                    Panel(
                        content,
                        title="Volume",
                        border_style=bias_color.lower(),
                        height=6,
                    )
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
                price_prim = Price(value=Decimal(str(price)))
                row = Text(
                    f"  ▶ {price_prim.format_display()} ◀  {level_text}",
                    style="bold yellow",
                )
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
            return Panel("No trade data...", title="Trade Momentum", border_style="dim")

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

        # If no time-based data, show total summary
        if not rows and self.trades:
            total_trades = len(self.trades)
            buy_count = sum(1 for t in self.trades if t.side.value == "buy")
            sell_count = total_trades - buy_count
            rows.append(Text(f"Total: {total_trades} trades", style="bold"))
            rows.append(Text(f"Buys: {buy_count} | Sells: {sell_count}"))

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
            Group(*rows) if rows else Text("Processing trades...", style="dim"),
            title="Trade Momentum",
            border_style="blue",
            padding=0,
        )


class MarketHealthWidget(Widget):
    """Market health metrics focusing on quality, liquidity, and spreads."""

    ticker: reactive[MarketTicker | None] = reactive(None)
    order_book: reactive[OrderBook | None] = reactive(None)

    def render(self) -> RenderableType:
        """Render market health assessment."""
        if not self.ticker or not self.order_book:
            return Panel("Awaiting data...", title="Market Health", border_style="dim")

        rows = []

        # Quality Score with visual indicator
        quality_score = self._calculate_quality_score()
        quality_color = (
            "green"
            if quality_score >= 80
            else "yellow"
            if quality_score >= 60
            else "red"
        )

        # Visual quality meter with gradient colors
        meter_width = 20
        filled = int((quality_score / 100) * meter_width)

        # Create gradient meter
        meter = Text()
        for i in range(meter_width):
            if i < filled:
                # Gradient from red to yellow to green
                if i < meter_width * 0.4:
                    meter.append("█", style="red")
                elif i < meter_width * 0.7:
                    meter.append("█", style="yellow")
                else:
                    meter.append("█", style="green")
            else:
                meter.append("░", style="dim")

        # Quality Score header
        header = Text()
        header.append("Quality Score", style="bold cyan")
        rows.append(header)

        # Quality meter and value
        quality_line = Text()
        quality_line.append(meter)
        quality_line.append(" ")
        quality_line.append(f"{quality_score:.0f}%", style=f"bold {quality_color}")
        rows.append(quality_line)
        rows.append(Text())

        # Liquidity Assessment with icons
        if self.ticker.volume is not None:
            vol_btc = float(self.ticker.volume)
            if vol_btc > 5000:
                liquidity_level = "High"
                liquidity_color = "green"
            elif vol_btc > 1000:
                liquidity_level = "Medium"
                liquidity_color = "yellow"
            else:
                liquidity_level = "Low"
                liquidity_color = "red"

            liq_text = Text()
            liq_text.append("Liquidity", style="bold white")
            liq_text.append(":", style="dim")
            liq_text.append(" ")
            liq_text.append(liquidity_level, style=f"bold {liquidity_color}")
            liq_text.append(" ")
            liq_text.append(f"({vol_btc:,.0f} BTC/24h)", style=f"{liquidity_color} dim")
            rows.append(liq_text)

        # Spread Analysis with visual indicator
        if self.order_book.spread_primitive:
            spread_bps = float(self.order_book.spread_primitive.as_basis_points().value)
            spread_usd = float(self.order_book.spread_primitive.value)

            # Visual spread indicator
            if spread_bps < 5:
                spread_quality = "Tight"
                spread_color = "green"
            elif spread_bps < 10:
                spread_quality = "Normal"
                spread_color = "yellow"
            else:
                spread_quality = "Wide"
                spread_color = "red"

            spread_text = Text()
            spread_text.append("Spread", style="bold white")
            spread_text.append(":", style="dim")
            spread_text.append(" ")
            spread_text.append(spread_quality, style=f"bold {spread_color}")
            spread_text.append(" ")
            spread_text.append(
                f"({spread_bps:.1f}bps / ${spread_usd:.2f})",
                style=f"{spread_color} dim",
            )
            rows.append(spread_text)

        # Book Balance with visual bar
        if self.order_book.bid_levels and self.order_book.ask_levels:
            bid_depth = sum(level.size for level in self.order_book.bid_levels[:5])
            ask_depth = sum(level.size for level in self.order_book.ask_levels[:5])
            total_depth = bid_depth + ask_depth

            if total_depth > 0:
                imbalance = float((bid_depth - ask_depth) / total_depth)
                bid_pct = float(bid_depth / total_depth)

                # Visual balance bar
                bar_width = 10
                bid_width = int(bid_pct * bar_width)
                ask_width = bar_width - bid_width

                balance_bar = Text()
                balance_bar.append("█" * bid_width, style="green")
                balance_bar.append("█" * ask_width, style="red")

                if abs(imbalance) < 0.2:
                    balance_text = "Balanced"
                    balance_color = "yellow"
                elif imbalance > 0:
                    balance_text = "Bid Heavy"
                    balance_color = "green"
                else:
                    balance_text = "Ask Heavy"
                    balance_color = "red"

                book_text = Text()
                book_text.append("Book", style="bold white")
                book_text.append(":", style="dim")
                book_text.append(" ")
                book_text.append(balance_text, style=f"bold {balance_color}")
                book_text.append(" ")
                book_text.append(f"({imbalance:+.0%})", style=f"{balance_color} dim")
                rows.append(book_text)

        return Panel(
            Group(*rows),
            title="Market Health",
            border_style=quality_color.lower(),
            padding=(0, 1),
        )

    def _calculate_quality_score(self) -> float:
        """Calculate overall market quality score."""
        score = 50.0  # Start with float

        # Volume component
        if self.ticker and self.ticker.volume:
            vol_score = min(30.0, float(self.ticker.volume) / 100)
            score += vol_score

        # Spread component
        if self.order_book and self.order_book.spread_primitive:
            spread_bps = float(self.order_book.spread_primitive.as_basis_points().value)
            spread_score = max(0.0, 20.0 - spread_bps)
            score += spread_score

        return min(100.0, score)


class TradeDynamicsWidget(Widget):
    """Trade dynamics showing momentum, volatility, and volume profile."""

    trades: reactive[list[MarketTrade]] = reactive(list)
    order_book: reactive[OrderBook | None] = reactive(None)
    indicators: reactive[list[BaseIndicator]] = reactive(list)

    def render(self) -> RenderableType:
        """Render trade dynamics analysis."""
        rows = []

        # Trade Momentum with enhanced visuals
        if self.trades:
            buy_volume = sum(float(t.size) for t in self.trades[:50] if t.is_buy)
            sell_volume = sum(float(t.size) for t in self.trades[:50] if t.is_sell)
            total_volume = buy_volume + sell_volume

            if total_volume > 0:
                buy_pct = buy_volume / total_volume
                momentum = (buy_pct - 0.5) * 200

                # Enhanced visual momentum bar with gradient
                bar_width = 20
                center = bar_width // 2

                momentum_bar = Text()
                for i in range(bar_width):
                    if i < center:
                        # Left side (sell pressure)
                        if momentum < 0:
                            intensity = abs(momentum) / 100
                            filled_pos = center - int(intensity * center)
                            if i >= filled_pos:
                                momentum_bar.append("█", style="red bold")
                            else:
                                momentum_bar.append("░", style="dim")
                        else:
                            momentum_bar.append("░", style="dim")
                    else:
                        # Right side (buy pressure)
                        if momentum > 0:
                            intensity = momentum / 100
                            filled_pos = center + int(intensity * center)
                            if i < filled_pos:
                                momentum_bar.append("█", style="green bold")
                            else:
                                momentum_bar.append("░", style="dim")
                        else:
                            momentum_bar.append("░", style="dim")

                # Momentum header
                mom_color = (
                    "green" if momentum > 20 else "red" if momentum < -20 else "yellow"
                )

                header = Text()
                header.append("Trade Momentum", style="bold cyan")
                rows.append(header)
                rows.append(momentum_bar)

                mom_text = Text()
                mom_text.append(f"{momentum:+.0f}%", style=f"bold {mom_color}")
                mom_text.append(" ")
                mom_text.append("Buy", style="dim")
                mom_text.append(":", style="dim")
                mom_text.append(" ")
                mom_text.append(f"{buy_pct:.0%}", style=f"{mom_color} dim")
                rows.append(mom_text)
                rows.append(Text())

        # Volatility with visual meter
        if self.order_book and self.order_book.spread_primitive:
            spread_bps = float(self.order_book.spread_primitive.as_basis_points().value)
            volatility = min(100, spread_bps * 5)

            # Visual volatility meter
            vol_meter = Text()
            meter_width = 15
            filled = int((volatility / 100) * meter_width)

            for i in range(meter_width):
                if i < filled:
                    if volatility < 30:
                        vol_meter.append("▪", style="green")
                    elif volatility < 70:
                        vol_meter.append("▪", style="yellow")
                    else:
                        vol_meter.append("▪", style="red bold")
                else:
                    vol_meter.append("·", style="dim")

            vol_level = (
                "Low" if volatility < 30 else "Medium" if volatility < 70 else "High"
            )
            vol_color = (
                "green" if volatility < 30 else "yellow" if volatility < 70 else "red"
            )

            vol_text = Text()
            vol_text.append("Volatility", style="bold white")
            vol_text.append(":", style="dim")
            vol_text.append(" ")
            vol_text.append(vol_level, style=f"bold {vol_color}")
            vol_text.append(" ")
            vol_text.append(f"({volatility:.0f}%)", style=f"{vol_color} dim")
            rows.append(vol_text)
            rows.append(vol_meter)

        # Volume Analysis with enhanced display
        from src.market.analysis.volume_analysis import VolumeAnalysis

        for ind in self.indicators:
            if isinstance(ind, VolumeAnalysis):
                rows.append(Text())

                # Volume ratio with visual indicator
                vol_ratio = float(ind.volume_ratio)
                ratio_color = (
                    "green"
                    if vol_ratio > 1.5
                    else "yellow"
                    if vol_ratio > 0.8
                    else "red"
                )

                vol_text = Text()
                vol_text.append("Vol Ratio", style="bold white")
                vol_text.append(":", style="dim")
                vol_text.append(" ")
                vol_text.append(f"{vol_ratio:.1f}x avg", style=f"bold {ratio_color}")
                rows.append(vol_text)

                # VWAP with color based on current price
                vwap_text = Text()
                vwap_text.append("VWAP", style="bold white")
                vwap_text.append(":", style="dim")
                vwap_text.append(" ")
                try:
                    vwap_value = float(ind.vwap)
                    if not (vwap_value != vwap_value):  # Check for NaN
                        vwap_text.append(f"${vwap_value:,.0f}", style="cyan")
                    else:
                        vwap_text.append("N/A", style="dim")
                except (ValueError, TypeError, AttributeError):
                    vwap_text.append("N/A", style="dim")
                rows.append(vwap_text)
                break

        # Dynamic panel color based on overall momentum
        panel_color = "cyan"
        if self.trades:
            # Recalculate for panel color
            buy_volume = sum(float(t.size) for t in self.trades[:50] if t.is_buy)
            sell_volume = sum(float(t.size) for t in self.trades[:50] if t.is_sell)
            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                buy_pct = buy_volume / total_volume
                momentum = (buy_pct - 0.5) * 200
                panel_color = (
                    "green" if momentum > 20 else "red" if momentum < -20 else "yellow"
                )

        return Panel(
            Group(*rows) if rows else Text("Analyzing trade flow...", style="dim"),
            title="Trade Dynamics",
            border_style=panel_color.lower(),
            padding=(0, 1),
        )


class TradingSignalsWidget(Widget):
    """Trading signals and alerts based on market conditions."""

    ticker: reactive[MarketTicker | None] = reactive(None)
    order_book: reactive[OrderBook | None] = reactive(None)
    trades: reactive[list[MarketTrade]] = reactive(list)
    indicators: reactive[list[BaseIndicator]] = reactive(list)

    def render(self) -> RenderableType:
        """Render trading signals and alerts."""
        rows = []

        # Generate signal
        signal, confidence = self._generate_signal()

        if signal != "HOLD":
            signal_color = "green" if signal == "BUY" else "red"

            # Big visual signal
            arrow = "▲" if signal == "BUY" else "▼"
            arrow_text = Text()
            arrow_text.append(arrow * 5, style=f"{signal_color} bold")
            arrow_text.justify = "center"
            rows.append(arrow_text)

            signal_text = Text()
            signal_text.append(signal, style=f"{signal_color} bold")
            signal_text.append(f" ({confidence:.0f}% conf)")
            signal_text.justify = "center"
            rows.append(signal_text)
            rows.append(Text())

            # Entry checklist
            rows.append(Text("Entry Conditions:", style="bold"))
            conditions = self._check_conditions()
            for condition, met in conditions:
                check = "✓" if met else "✗"
                color = "green" if met else "red"
                condition_text = Text()
                condition_text.append(f"  {check} ", style=color)
                condition_text.append(condition)
                rows.append(condition_text)
        else:
            neutral_text = Text()
            neutral_text.append("⟷ NEUTRAL", style="yellow bold")
            neutral_text.justify = "center"
            rows.append(neutral_text)

            waiting_text = Text()
            waiting_text.append("Waiting for setup...", style="dim")
            waiting_text.justify = "center"
            rows.append(waiting_text)

        # Alerts
        alerts = self._generate_alerts()
        if alerts:
            rows.append(Text())
            rows.append(Text("Alerts:", style="bold yellow"))
            for alert in alerts[:2]:
                alert_text = Text()
                alert_text.append("  ⚡ ", style="yellow")
                alert_text.append(alert, style="yellow")
                rows.append(alert_text)

        panel_color = (
            "green" if signal == "BUY" else "red" if signal == "SELL" else "yellow"
        )

        return Panel(
            Group(*rows),
            title="Trading Signals",
            border_style=panel_color.lower(),
            padding=(0, 1),
        )

    def _generate_signal(self) -> tuple[str, float]:
        """Generate trading signal based on conditions."""
        conditions_met = 0
        total_conditions = 0

        # Check spread
        if self.order_book and self.order_book.spread_primitive:
            total_conditions += 1
            if float(self.order_book.spread_primitive.as_basis_points().value) < 10:
                conditions_met += 1

        # Check momentum
        if self.trades:
            total_conditions += 1
            buy_vol = sum(float(t.size) for t in self.trades[:20] if t.is_buy)
            sell_vol = sum(float(t.size) for t in self.trades[:20] if t.is_sell)
            if buy_vol > sell_vol * 1.5:
                conditions_met += 1
                direction = "BUY"
            elif sell_vol > buy_vol * 1.5:
                conditions_met += 1
                direction = "SELL"
            else:
                direction = "HOLD"
        else:
            direction = "HOLD"

        # Check indicators
        from src.market.analysis.momentum_indicators import RSIAnalysis

        for ind in self.indicators:
            if isinstance(ind, RSIAnalysis):
                total_conditions += 1
                if ind.rsi_value < 30:
                    conditions_met += 1
                    direction = "BUY"
                elif ind.rsi_value > 70:
                    conditions_met += 1
                    direction = "SELL"

        confidence = (
            (conditions_met / total_conditions * 100) if total_conditions > 0 else 50
        )

        if conditions_met >= 2 and direction != "HOLD":
            return direction, confidence

        return "HOLD", 50

    def _check_conditions(self) -> list[tuple[str, bool]]:
        """Check entry conditions."""
        conditions = []

        # Spread condition
        if self.order_book and self.order_book.spread_primitive:
            spread_bps = float(self.order_book.spread_primitive.as_basis_points().value)
            conditions.append(("Tight spread (<10bps)", spread_bps < 10))

        # Volume condition
        if self.ticker and self.ticker.volume:
            conditions.append(("High liquidity", float(self.ticker.volume) > 1000))

        # Momentum alignment
        if self.trades:
            buy_vol = sum(float(t.size) for t in self.trades[:20] if t.is_buy)
            sell_vol = sum(float(t.size) for t in self.trades[:20] if t.is_sell)
            conditions.append(
                (
                    "Strong momentum",
                    abs(buy_vol - sell_vol) > (buy_vol + sell_vol) * 0.2,
                )
            )

        return conditions

    def _generate_alerts(self) -> list[str]:
        """Generate trading alerts."""
        alerts = []

        # Check indicators
        from src.market.analysis.macd import MACDAnalysis
        from src.market.analysis.momentum_indicators import RSIAnalysis

        for ind in self.indicators:
            if isinstance(ind, RSIAnalysis):
                if ind.rsi_value > 70:
                    alerts.append(f"RSI overbought ({float(ind.rsi_value):.0f})")
                elif ind.rsi_value < 30:
                    alerts.append(f"RSI oversold ({float(ind.rsi_value):.0f})")
            elif isinstance(ind, MACDAnalysis) and ind.crossover_detected:
                alerts.append(f"MACD {ind.crossover_type} cross")

        return alerts


class DenseMarketDashboard(App[None]):
    """Market Service - AI-powered trading analysis dashboard."""

    CSS = """
    Screen {
        background: $surface;
    }
    
    #main-content {
        height: 14;
        width: 100%;
    }
    
    #orderbook {
        width: 40%;
        height: 14;
    }
    
    #trades {
        width: 30%;
        height: 14;
    }
    
    #profile {
        width: 30%;
        height: 14;
    }
    
    #middle-row {
        height: 12;
        width: 100%;
    }
    
    #health {
        width: 33%;
        height: 12;
    }
    
    #dynamics {
        width: 34%;
        height: 12;
    }
    
    #signals {
        width: 33%;
        height: 12;
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

        # Middle row with three focused panels
        with Horizontal(id="middle-row"):
            yield MarketHealthWidget(id="health")
            yield TradeDynamicsWidget(id="dynamics")
            yield TradingSignalsWidget(id="signals")

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
        # Use live data from Coinbase
        from src.market.service.market_stream import stream_market_data

        def on_snapshot_callback(snapshot: MarketSnapshot) -> None:
            """Handle snapshot callback from WebSocket thread."""
            # Check if app is still running before calling from thread
            if self.is_running:
                try:
                    # Use Textual's call_from_thread to properly schedule the update
                    self.call_from_thread(self.process_snapshot, snapshot)
                except Exception:
                    # App is shutting down or error occurred, ignore
                    pass

        self.ws_client = stream_market_data(
            symbols=[self.current_symbol],
            on_snapshot=on_snapshot_callback,
            exchange="coinbase",
        )

        self.notify(
            f"Connected to Coinbase for {self.current_symbol}", severity="information"
        )

    def process_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Process snapshot in the main thread."""
        # Check if app is still running before scheduling task
        if self.is_running:
            try:
                # Create async task in the app's event loop
                _ = asyncio.create_task(self.handle_market_snapshot(snapshot))  # noqa: RUF006
            except RuntimeError:
                # App is shutting down, ignore
                pass

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
                float(spread / ((snapshot.ticker.bid + snapshot.ticker.ask) / 2))
                * 10000
                if snapshot.ticker.bid and snapshot.ticker.ask
                else 0
            )

            # Use domain primitives for rich formatting
            price_display = snapshot.ticker.price_primitive.format_display()
            spread_display = f"{spread_bps:.0f}bps" if spread_bps else "N/A"
            volume_display = (
                snapshot.ticker.format_volume() if snapshot.ticker.volume else "N/A"
            )

            self.sub_title = (
                f"{price_display} | Spread: {spread_display} | Vol: {volume_display}"
            )

        # Update widgets
        if snapshot.order_book and isinstance(snapshot.order_book, OrderBook):
            self.query_one(
                "#orderbook", OrderBookDepthWidget
            ).order_book = snapshot.order_book
            self.query_one(
                "#profile", MarketProfileWidget
            ).order_book = snapshot.order_book
            self.query_one(
                "#health", MarketHealthWidget
            ).order_book = snapshot.order_book
            self.query_one(
                "#dynamics", TradeDynamicsWidget
            ).order_book = snapshot.order_book
            self.query_one(
                "#signals", TradingSignalsWidget
            ).order_book = snapshot.order_book

        if snapshot.ticker and isinstance(snapshot.ticker, MarketTicker):
            self.query_one("#profile", MarketProfileWidget).ticker = snapshot.ticker
            self.query_one("#health", MarketHealthWidget).ticker = snapshot.ticker
            self.query_one("#signals", TradingSignalsWidget).ticker = snapshot.ticker

        if snapshot.trades:
            trades_widget = self.query_one("#trades", TradeMomentumWidget)
            current_trades = list(trades_widget.trades)
            new_trades = [t for t in snapshot.trades if isinstance(t, MarketTrade)]
            trades_widget.trades = new_trades + current_trades[:100]

            # Also update dynamics and signals widgets with trades
            self.query_one(
                "#dynamics", TradeDynamicsWidget
            ).trades = trades_widget.trades
            self.query_one(
                "#signals", TradingSignalsWidget
            ).trades = trades_widget.trades

        # Update price series
        await self.analysis_service._update_price_series(snapshot.symbol, snapshot)

    async def update_indicators(self) -> None:
        """Update technical indicators."""
        while self.is_running:
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

                    if result.success and result.indicators and self.is_running:
                        self.query_one(
                            "#indicators", CompactIndicatorGrid
                        ).indicators = result.indicators
                        self.query_one(
                            "#profile", MarketProfileWidget
                        ).indicators = result.indicators
                        self.query_one(
                            "#dynamics", TradeDynamicsWidget
                        ).indicators = result.indicators
                        self.query_one(
                            "#signals", TradingSignalsWidget
                        ).indicators = result.indicators

            except asyncio.CancelledError:
                # Task was cancelled, exit cleanly
                break
            except Exception as e:
                if self.is_running:
                    self.notify(f"Indicator error: {e}", severity="warning")

            await asyncio.sleep(2)  # Faster updates

    async def on_unmount(self) -> None:
        """Clean up when app unmounts."""
        # Cancel update task
        if self.update_task and not self.update_task.done():
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass

        # Don't try to close WebSocket during shutdown - it blocks indefinitely
        # The WebSocket will be cleaned up when the process exits
        # Just clear the reference to help garbage collection
        self.ws_client = None

    async def action_toggle_symbol(self) -> None:
        """Switch to next symbol."""
        current_idx = self.symbols.index(self.current_symbol)
        next_idx = (current_idx + 1) % len(self.symbols)
        self.current_symbol = self.symbols[next_idx]

        self.title = f"Market Service - {self.current_symbol}"
        self.notify(f"Switched to {self.current_symbol}")

        # Clear the old WebSocket reference instead of calling close()
        # which blocks indefinitely
        self.ws_client = None

        # Give garbage collector a moment to clean up
        await asyncio.sleep(0.2)

        # Restart with new symbol
        await self.start_market_stream()


def run_dashboard() -> None:
    """Run the information-dense market dashboard."""
    app = DenseMarketDashboard()
    app.run()


if __name__ == "__main__":
    run_dashboard()
