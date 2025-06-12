"""
Analysis pipeline demonstrating composed Pydantic models.

This shows how to chain analysis models together:
MarketSnapshot → SpreadAnalysis → MarketMicrostructure → TradingSignal
                → TradeFlowAnalysis ↗

Each stage validates its inputs and outputs using Pydantic.
"""

from collections import deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from src.market.analysis.models import (
    MarketMicrostructure,
    SpreadAnalysis,
    TradeFlowAnalysis,
    TradingSignal,
)
from src.market.model.snapshot import MarketSnapshot


class AnalysisPipeline:
    """
    Orchestrates the flow of data through analysis models.

    This demonstrates the "composed Pydantic model machines" concept:
    - Maintains history for time-series analysis
    - Chains models together with validated inputs/outputs
    - Produces actionable trading signals
    """

    def __init__(self, history_minutes: int = 5) -> None:
        """
        Initialize pipeline with configurable history window.

        Args:
            history_minutes: How many minutes of history to maintain

        """
        self.history_minutes = history_minutes
        self.snapshot_history: dict[str, deque[MarketSnapshot]] = {}
        self.analysis_cache: dict[str, MarketMicrostructure] = {}

    def process_snapshot(self, snapshot: MarketSnapshot) -> TradingSignal | None:
        """
        Process a market snapshot through the full analysis pipeline.

        Args:
            snapshot: Latest market snapshot

        Returns:
            Trading signal if analysis can be performed, None otherwise

        """
        # Update history
        self._update_history(snapshot)

        # Stage 1: Analyze spread patterns
        spread_analysis = self._analyze_spread(snapshot)
        if not spread_analysis:
            return None

        # Stage 2: Analyze trade flow
        trade_flow = self._analyze_trade_flow(snapshot)
        if not trade_flow:
            return None

        # Stage 3: Combine into microstructure analysis
        microstructure = MarketMicrostructure.from_analyses(
            symbol=snapshot.symbol,
            spread_analysis=spread_analysis,
            trade_flow=trade_flow,
        )

        # Cache for potential cross-symbol analysis
        self.analysis_cache[snapshot.symbol] = microstructure

        # Stage 4: Generate trading signal
        current_price = snapshot.ticker.price if snapshot.ticker else Decimal("0")
        if current_price > 0:
            signal = TradingSignal.from_microstructure(
                microstructure=microstructure,
                current_price=current_price,
            )
            return signal

        return None

    def _update_history(self, snapshot: MarketSnapshot) -> None:
        """Maintain rolling history window for each symbol."""
        if snapshot.symbol not in self.snapshot_history:
            self.snapshot_history[snapshot.symbol] = deque()

        history = self.snapshot_history[snapshot.symbol]
        history.append(snapshot)

        # Remove old snapshots
        cutoff = datetime.now(UTC) - timedelta(minutes=self.history_minutes)
        while history and history[0].timestamp < cutoff:
            history.popleft()

    def _analyze_spread(self, snapshot: MarketSnapshot) -> SpreadAnalysis | None:
        """Create spread analysis with historical context."""
        if not snapshot.order_book:
            return None

        history = list(self.snapshot_history.get(snapshot.symbol, []))

        try:
            return SpreadAnalysis.from_snapshot(
                snapshot=snapshot,
                history=history[:-1],  # Exclude current snapshot
            )
        except ValueError:
            # Handle missing data gracefully
            return None

    def _analyze_trade_flow(self, snapshot: MarketSnapshot) -> TradeFlowAnalysis | None:
        """Create trade flow analysis from snapshot."""
        try:
            return TradeFlowAnalysis.from_snapshot(snapshot)
        except Exception:
            # Handle missing data gracefully
            return None

    def get_market_summary(self) -> dict[str, MarketMicrostructure]:
        """Get current market microstructure for all tracked symbols."""
        return self.analysis_cache.copy()


# Example usage demonstrating the pipeline
def example_pipeline_usage() -> tuple[Any, AnalysisPipeline]:
    """Show how to use the analysis pipeline with streaming data."""
    from src.market.adapters.coinbase.stream import stream_coinbase_market_data

    # Create pipeline
    pipeline = AnalysisPipeline(history_minutes=5)

    def handle_snapshot(snapshot: MarketSnapshot) -> None:
        """Process each snapshot through the pipeline."""
        # Run analysis pipeline
        signal = pipeline.process_snapshot(snapshot)

        if signal:
            # Log the signal (in production, this would go to a trading system)
            print(f"\n📊 Trading Signal for {signal.symbol}")
            print(
                f"  Signal: {signal.signal_type.upper()} "
                f"(confidence: {signal.confidence}%)"
            )
            print(f"  Reason: {signal.reason}")
            print(f"  Market Quality: {signal.microstructure.market_quality}")
            print(f"  Execution Cost: {signal.microstructure.execution_cost_bps} bps")

            if signal.signal_type != "hold":
                print(f"  Max Position: {signal.max_position_size}")
                if signal.stop_loss_price:
                    print(f"  Stop Loss: ${signal.stop_loss_price}")
                if signal.take_profit_price:
                    print(f"  Take Profit: ${signal.take_profit_price}")

            # Show component analyses
            print("\n  Spread Analysis:")
            print(
                f"    Current: {signal.microstructure.spread_analysis.spread_bps} bps"
            )
            print(f"    Widening: {signal.microstructure.spread_analysis.is_widening}")
            print(
                f"    Volatility: "
                f"{signal.microstructure.spread_analysis.volatility_score}/100"
            )

            print("\n  Trade Flow:")
            print(f"    Momentum: {signal.microstructure.trade_flow.momentum_score}")
            print(f"    Buy Pressure: {signal.microstructure.trade_flow.buy_pressure}%")
            print(f"    VWAP: ${signal.microstructure.trade_flow.vwap}")

    # Stream data through pipeline
    print("🚀 Starting analysis pipeline...")
    client = stream_coinbase_market_data(
        symbols=["BTC-USD", "ETH-USD"],
        on_snapshot=handle_snapshot,
    )

    return client, pipeline


if __name__ == "__main__":
    # Run example when module is executed directly
    client, pipeline = example_pipeline_usage()

    try:
        # Run for a while
        import time

        time.sleep(60)  # Run for 1 minute

        # Show final market summary
        print("\n📈 Market Summary:")
        for symbol, micro in pipeline.get_market_summary().items():
            print(f"\n{symbol}:")
            print(f"  Market Quality: {micro.market_quality}")
            print(f"  Liquidity Score: {micro.liquidity_score}")
            entry_conditions = micro.analyze_entry_conditions()
            print(f"  Favorable Entry: {entry_conditions['all_conditions_met']}")

    finally:
        client.close()
        print("\n✅ Pipeline stopped")
