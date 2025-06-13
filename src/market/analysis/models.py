"""
Composed Pydantic analysis models for market data processing.

These models demonstrate the "composed pydantic model machines" pattern:
- Each model takes structured input (snapshots or other models)
- Performs specific analysis/calculations
- Outputs structured, validated data
- Can be chained together to form analysis pipelines
"""

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

from src.market.analysis.contexts import (
    MicrostructureAgentContext,
    SpreadInfo,
    TradeFlowInfo,
)
from src.market.model.snapshot import MarketSnapshot


class SpreadAnalysis(BaseModel):
    """
    Analyzes spread behavior over time.

    Takes a snapshot and recent history to identify spread patterns.
    """

    symbol: str
    timestamp: datetime
    current_spread: Decimal
    spread_percentage: Decimal
    spread_bps: Decimal = Field(description="Spread in basis points")

    # Historical comparisons
    avg_spread_5min: Decimal | None = None
    spread_percentile: Decimal = Field(
        default=Decimal("50"),
        description="Where current spread falls in recent distribution (0-100)",
    )
    is_widening: bool = False
    volatility_score: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="0=stable, 100=extremely volatile",
    )

    @classmethod
    def from_snapshot(
        cls, snapshot: MarketSnapshot, history: list[MarketSnapshot] | None = None
    ) -> "SpreadAnalysis":
        """Create spread analysis from current snapshot and optional history."""
        if not snapshot.order_book:
            raise ValueError("Snapshot must have order book data")

        current_spread = snapshot.order_book.spread or Decimal("0")
        spread_pct = snapshot.order_book.spread_percentage or Decimal("0")
        spread_bps = spread_pct * 100  # Convert percentage to basis points

        # If we have history, calculate trends
        avg_spread = None
        is_widening = False
        volatility_score = Decimal("0")

        if history and len(history) >= 2:
            # Calculate average spread
            spreads = [
                h.order_book.spread
                for h in history
                if h.order_book and h.order_book.spread
            ]
            if spreads:
                avg_spread = sum(spreads, Decimal("0")) / Decimal(str(len(spreads)))
                is_widening = current_spread > avg_spread

                # Simple volatility score based on spread variance
                if len(spreads) > 1:
                    mean = sum(spreads, Decimal("0")) / Decimal(str(len(spreads)))
                    variance = sum((s - mean) ** 2 for s in spreads) / Decimal(
                        str(len(spreads))
                    )
                    volatility_score = min(Decimal("100"), variance * Decimal("1000"))

        return cls(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            current_spread=current_spread,
            spread_percentage=spread_pct,
            spread_bps=spread_bps,
            avg_spread_5min=avg_spread,
            is_widening=is_widening,
            volatility_score=volatility_score,
        )


class TradeFlowAnalysis(BaseModel):
    """
    Analyzes trade flow and momentum from recent trades.

    Identifies buying/selling pressure and price momentum.
    """

    symbol: str
    timestamp: datetime
    trade_count: int
    total_volume: Decimal

    # Directional analysis
    buy_volume: Decimal
    sell_volume: Decimal
    net_flow: Decimal = Field(description="Buy volume - sell volume")
    buy_pressure: Decimal = Field(
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Buy volume as percentage of total",
    )

    # Price momentum
    vwap: Decimal = Field(description="Volume weighted average price")
    price_trend: Literal["up", "down", "neutral"]
    momentum_score: Decimal = Field(
        description="Positive=bullish, negative=bearish",
        ge=Decimal("-100"),
        le=Decimal("100"),
    )

    @classmethod
    def from_snapshot(cls, snapshot: MarketSnapshot) -> "TradeFlowAnalysis":
        """Analyze trade flow from snapshot's recent trades."""
        if not snapshot.trades:
            # Return neutral analysis if no trades
            return cls(
                symbol=snapshot.symbol,
                timestamp=snapshot.timestamp,
                trade_count=0,
                total_volume=Decimal("0"),
                buy_volume=Decimal("0"),
                sell_volume=Decimal("0"),
                net_flow=Decimal("0"),
                buy_pressure=Decimal("50"),
                vwap=snapshot.ticker.price if snapshot.ticker else Decimal("0"),
                price_trend="neutral",
                momentum_score=Decimal("0"),
            )

        # Calculate volumes by side
        buy_volume = sum(
            (t.size for t in snapshot.trades if t.side.value == "buy"), Decimal("0")
        )
        sell_volume = sum(
            (t.size for t in snapshot.trades if t.side.value == "sell"), Decimal("0")
        )
        total_volume = buy_volume + sell_volume

        # Calculate VWAP
        if total_volume > 0:
            total_value = sum(t.price * t.size for t in snapshot.trades)
            vwap = total_value / total_volume
            buy_pressure = (buy_volume / total_volume) * Decimal("100")
        else:
            vwap = snapshot.trades[0].price if snapshot.trades else Decimal("0")
            buy_pressure = Decimal("50")

        # Determine trend
        net_flow = buy_volume - sell_volume
        price_trend: Literal["up", "down", "neutral"]
        if buy_pressure > 60:
            price_trend = "up"
            momentum_score = min(
                Decimal("100"), (buy_pressure - Decimal("50")) * Decimal("2")
            )
        elif buy_pressure < 40:
            price_trend = "down"
            momentum_score = max(
                Decimal("-100"), (buy_pressure - Decimal("50")) * Decimal("2")
            )
        else:
            price_trend = "neutral"
            momentum_score = Decimal("0")

        return cls(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            trade_count=len(snapshot.trades),
            total_volume=total_volume,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            net_flow=net_flow,
            buy_pressure=buy_pressure,
            vwap=vwap,
            price_trend=price_trend,
            momentum_score=momentum_score,
        )


class MarketMicrostructure(BaseModel):
    """
    Analyzes market microstructure combining order book and trade data.

    This is a higher-level analysis that combines multiple inputs.
    """

    symbol: str
    timestamp: datetime

    # Component analyses
    spread_analysis: SpreadAnalysis
    trade_flow: TradeFlowAnalysis

    # Derived insights
    liquidity_score: Decimal = Field(
        ge=Decimal("0"), le=Decimal("100"), description="0=illiquid, 100=highly liquid"
    )
    market_quality: Literal["excellent", "good", "fair", "poor"]
    execution_cost_bps: Decimal = Field(
        description="Estimated cost to execute in basis points"
    )

    def analyze_entry_conditions(
        self,
        max_spread_bps: Decimal = Decimal("10"),
        min_liquidity_score: Decimal = Decimal("70"),
        acceptable_qualities: list[str] | None = None,
    ) -> dict[str, bool | Decimal | str]:
        """
        Analyze entry conditions with configurable thresholds.

        This provides detailed analysis for agents to make decisions,
        rather than making the decision itself.

        Args:
            max_spread_bps: Maximum acceptable spread in basis points
            min_liquidity_score: Minimum acceptable liquidity score
            acceptable_qualities: List of acceptable market qualities

        Returns:
            Dictionary with entry condition analysis

        """
        if acceptable_qualities is None:
            acceptable_qualities = ["excellent", "good"]

        return {
            "spread_acceptable": self.spread_analysis.spread_bps <= max_spread_bps,
            "spread_bps": self.spread_analysis.spread_bps,
            "spread_threshold": max_spread_bps,
            "liquidity_acceptable": self.liquidity_score >= min_liquidity_score,
            "liquidity_score": self.liquidity_score,
            "liquidity_threshold": min_liquidity_score,
            "quality_acceptable": self.market_quality in acceptable_qualities,
            "market_quality": self.market_quality,
            "execution_cost_bps": self.execution_cost_bps,
            "volatility_score": self.spread_analysis.volatility_score,
            "all_conditions_met": (
                self.spread_analysis.spread_bps <= max_spread_bps
                and self.liquidity_score >= min_liquidity_score
                and self.market_quality in acceptable_qualities
            ),
        }

    def to_agent_context(self) -> MicrostructureAgentContext:
        """
        Format microstructure analysis for agent consumption.

        Provides rich analytical data without making decisions.
        """
        return MicrostructureAgentContext(
            symbol=self.symbol,
            timestamp=self.timestamp.isoformat(),
            market_quality=self.market_quality,
            liquidity_score=float(self.liquidity_score),
            execution_cost_bps=float(self.execution_cost_bps),
            spread=SpreadInfo(
                current_bps=float(self.spread_analysis.spread_bps),
                is_widening=self.spread_analysis.is_widening,
                volatility=float(self.spread_analysis.volatility_score),
            ),
            trade_flow=TradeFlowInfo(
                momentum_score=float(self.trade_flow.momentum_score),
                buy_pressure=float(self.trade_flow.buy_pressure),
                trend=self.trade_flow.price_trend,
                volume=float(self.trade_flow.total_volume),
            ),
            summary=self._generate_summary(),
        )

    def _generate_summary(self) -> str:
        """Generate a natural language summary of market conditions."""
        summaries = []

        # Market quality summary
        summaries.append(f"{self.market_quality.title()} market quality")

        # Liquidity summary
        if self.liquidity_score >= 80:
            summaries.append("highly liquid")
        elif self.liquidity_score >= 60:
            summaries.append("good liquidity")
        elif self.liquidity_score >= 40:
            summaries.append("moderate liquidity")
        else:
            summaries.append("low liquidity")

        # Spread conditions
        if self.spread_analysis.spread_bps < 5:
            summaries.append("tight spreads")
        elif self.spread_analysis.spread_bps < 10:
            summaries.append("normal spreads")
        else:
            summaries.append("wide spreads")

        # Momentum
        if abs(self.trade_flow.momentum_score) > 50:
            direction = "bullish" if self.trade_flow.momentum_score > 0 else "bearish"
            summaries.append(f"strong {direction} momentum")
        elif abs(self.trade_flow.momentum_score) > 20:
            direction = "bullish" if self.trade_flow.momentum_score > 0 else "bearish"
            summaries.append(f"{direction} momentum")

        return f"{self.symbol}: {', '.join(summaries)}"

    @classmethod
    def from_analyses(
        cls,
        symbol: str,
        spread_analysis: SpreadAnalysis,
        trade_flow: TradeFlowAnalysis,
    ) -> "MarketMicrostructure":
        """Combine multiple analyses into microstructure assessment."""
        # Calculate liquidity score based on spread and volume
        spread_score = max(0, 100 - float(spread_analysis.spread_bps))
        volume_score = min(100, float(trade_flow.total_volume) * 10)
        liquidity_score = Decimal((spread_score + volume_score) / 2)

        # Determine market quality
        market_quality: Literal["excellent", "good", "fair", "poor"]
        if spread_analysis.spread_bps < 5 and liquidity_score > 80:
            market_quality = "excellent"
        elif spread_analysis.spread_bps < 10 and liquidity_score > 60:
            market_quality = "good"
        elif spread_analysis.spread_bps < 20 and liquidity_score > 40:
            market_quality = "fair"
        else:
            market_quality = "poor"

        # Estimate execution cost (simplified)
        execution_cost = spread_analysis.spread_bps / 2  # Half spread
        if spread_analysis.volatility_score > 50:
            execution_cost *= Decimal("1.5")  # Add volatility premium

        return cls(
            symbol=symbol,
            timestamp=max(spread_analysis.timestamp, trade_flow.timestamp),
            spread_analysis=spread_analysis,
            trade_flow=trade_flow,
            liquidity_score=liquidity_score,
            market_quality=market_quality,
            execution_cost_bps=execution_cost,
        )


class TradingSignal(BaseModel):
    """
    Final trading signal combining all analyses.

    This demonstrates how analyses can be composed into actionable signals.
    """

    symbol: str
    timestamp: datetime
    microstructure: MarketMicrostructure

    # Signal properties
    signal_type: Literal["buy", "sell", "hold"]
    confidence: Decimal = Field(ge=Decimal("0"), le=Decimal("100"))
    reason: str

    # Risk metrics
    max_position_size: Decimal = Field(description="Maximum recommended position")
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None

    @classmethod
    def from_microstructure(
        cls,
        microstructure: MarketMicrostructure,
        current_price: Decimal,
    ) -> "TradingSignal":
        """Generate trading signal from market microstructure analysis."""
        # Simple signal logic for demonstration
        signal_type: Literal["buy", "sell", "hold"] = "hold"
        confidence = Decimal("50")
        reason = "Neutral market conditions"

        # Buy signal conditions
        entry_conditions = microstructure.analyze_entry_conditions()
        if (
            entry_conditions["all_conditions_met"]
            and microstructure.trade_flow.momentum_score > 20
            and microstructure.spread_analysis.is_widening is False
        ):
            signal_type = "buy"
            confidence = min(
                Decimal("90"),
                Decimal("50") + microstructure.trade_flow.momentum_score / 2,
            )
            reason = "Favorable entry with positive momentum"

        # Sell signal conditions
        elif (
            microstructure.trade_flow.momentum_score < -20
            and microstructure.spread_analysis.is_widening is True
        ):
            signal_type = "sell"
            confidence = min(
                Decimal("90"),
                Decimal("50") + abs(microstructure.trade_flow.momentum_score) / 2,
            )
            reason = "Negative momentum with widening spreads"

        # Calculate position sizing based on market quality
        if microstructure.market_quality == "excellent":
            max_position = Decimal("1.0")
        elif microstructure.market_quality == "good":
            max_position = Decimal("0.7")
        elif microstructure.market_quality == "fair":
            max_position = Decimal("0.4")
        else:
            max_position = Decimal("0.2")

        # Simple stop/target calculation
        stop_loss = None
        take_profit = None
        if signal_type == "buy":
            stop_loss = current_price * Decimal("0.98")  # 2% stop
            take_profit = current_price * Decimal("1.03")  # 3% target
        elif signal_type == "sell":
            stop_loss = current_price * Decimal("1.02")  # 2% stop
            take_profit = current_price * Decimal("0.97")  # 3% target

        return cls(
            symbol=microstructure.symbol,
            timestamp=microstructure.timestamp,
            microstructure=microstructure,
            signal_type=signal_type,
            confidence=confidence,
            reason=reason,
            max_position_size=max_position,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
        )
