"""Integration test for the analysis pipeline."""

from datetime import UTC, datetime
from decimal import Decimal

from src.market.analysis.models import (
    MarketMicrostructure,
    SpreadAnalysis,
    TradeFlowAnalysis,
    TradingSignal,
)
from src.market.analysis.pipeline import AnalysisPipeline
from src.market.model.snapshot import MarketSnapshot
from src.market.model.ticker import MarketTicker
from src.market.model.trade import MarketTrade
from tests.unit.app.market.helpers import OrderBookBuilder, TickerBuilder, TradeBuilder


class TestAnalysisPipeline:
    """Test the full analysis pipeline with composed models."""

    def test_pipeline_processes_snapshot_to_signal(self) -> None:
        """Test that pipeline can process a snapshot into a trading signal."""
        # Given: A pipeline
        pipeline = AnalysisPipeline(history_minutes=5)

        # And: A complete market snapshot
        ticker_data = TickerBuilder().with_price("50000.00").build()
        # Convert to domain model
        ticker = MarketTicker(
            symbol=ticker_data.symbol,
            price=ticker_data.price,
            bid=ticker_data.bid,
            ask=ticker_data.ask,
            volume=ticker_data.volume,
            timestamp=ticker_data.timestamp,
        )

        order_book = (
            OrderBookBuilder()
            .with_bid(49999, 1.0)
            .with_bid(49998, 2.0)
            .with_ask(50001, 1.0)
            .with_ask(50002, 2.0)
            .build()
        )

        trade_data_list = [
            TradeBuilder().with_price("50000").with_side("BUY").build(),
            TradeBuilder().with_price("49999").with_side("BUY").build(),
            TradeBuilder().with_price("50001").with_side("SELL").build(),
        ]
        # Convert to domain models
        trades = [
            MarketTrade(
                symbol=td.symbol,
                price=td.price,
                size=td.size,
                side=td.side,
                timestamp=td.timestamp,
                trade_id=td.trade_id,
            )
            for td in trade_data_list
        ]

        snapshot = MarketSnapshot(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            ticker=ticker,
            order_book=order_book,
            trades=trades,
        )

        # When: Processing the snapshot
        signal = pipeline.process_snapshot(snapshot)

        # Then: Should produce a trading signal
        assert signal is not None
        assert isinstance(signal, TradingSignal)
        assert signal.symbol == "BTC-USD"
        assert signal.signal_type in ["buy", "sell", "hold"]
        assert 0 <= signal.confidence <= 100

        # And: Signal should contain complete analysis chain
        assert isinstance(signal.microstructure, MarketMicrostructure)
        assert isinstance(signal.microstructure.spread_analysis, SpreadAnalysis)
        assert isinstance(signal.microstructure.trade_flow, TradeFlowAnalysis)

    def test_pipeline_builds_history(self) -> None:
        """Test that pipeline maintains history for analysis."""
        # Given: A pipeline
        pipeline = AnalysisPipeline(history_minutes=5)

        # When: Processing multiple snapshots
        for i in range(5):
            ticker_data = TickerBuilder().with_price(f"{50000 + i * 10}").build()
            # Convert to domain model
            ticker = MarketTicker(
                symbol=ticker_data.symbol,
                price=ticker_data.price,
                bid=ticker_data.bid,
                ask=ticker_data.ask,
                volume=ticker_data.volume,
                timestamp=ticker_data.timestamp,
            )
            snapshot = MarketSnapshot(
                symbol="BTC-USD",
                timestamp=datetime.now(UTC),
                ticker=ticker,
            )
            pipeline.process_snapshot(snapshot)

        # Then: History should be maintained
        assert "BTC-USD" in pipeline.snapshot_history
        assert len(pipeline.snapshot_history["BTC-USD"]) == 5

    def test_spread_analysis_calculation(self) -> None:
        """Test spread analysis calculations."""
        # Given: A snapshot with order book
        order_book = (
            OrderBookBuilder().with_bid(49990, 1.0).with_ask(50010, 1.0).build()
        )

        snapshot = MarketSnapshot(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            order_book=order_book,
        )

        # When: Creating spread analysis
        analysis = SpreadAnalysis.from_snapshot(snapshot)

        # Then: Calculations should be correct
        assert analysis.current_spread == Decimal("20")
        assert analysis.spread_bps == Decimal("4")  # 0.04% = 4 basis points

    def test_trade_flow_analysis_calculation(self) -> None:
        """Test trade flow analysis calculations."""
        # Given: A snapshot with trades
        trade_data_list = [
            TradeBuilder()
            .with_price("50000")
            .with_size("1.0")
            .with_side("BUY")
            .build(),
            TradeBuilder()
            .with_price("50100")
            .with_size("2.0")
            .with_side("BUY")
            .build(),
            TradeBuilder()
            .with_price("49900")
            .with_size("1.0")
            .with_side("SELL")
            .build(),
        ]
        # Convert to domain models
        trades = [
            MarketTrade(
                symbol=td.symbol,
                price=td.price,
                size=td.size,
                side=td.side,
                timestamp=td.timestamp,
                trade_id=td.trade_id,
            )
            for td in trade_data_list
        ]

        snapshot = MarketSnapshot(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            trades=trades,
        )

        # When: Creating trade flow analysis
        analysis = TradeFlowAnalysis.from_snapshot(snapshot)

        # Then: Calculations should be correct
        assert analysis.trade_count == 3
        assert analysis.total_volume == Decimal("4.0")
        assert analysis.buy_volume == Decimal("3.0")
        assert analysis.sell_volume == Decimal("1.0")
        assert analysis.buy_pressure == Decimal("75")  # 3/4 * 100
        assert analysis.price_trend == "up"
        assert analysis.momentum_score > 0
