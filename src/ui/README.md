# Market Analysis Dashboard

An information-dense terminal UI for real-time market analysis using Textual.

## Features

### Visual Data Density

- **Order Book Depth Visualization**: Visual bars showing bid/ask depth with imbalance metrics
- **Compact Indicator Grid**: All indicators visible at once with visual gauges
- **Market Profile**: Price levels with volume nodes and key support/resistance
- **Trade Flow Momentum**: Buy/sell pressure bars across multiple timeframes

### Key Metrics at a Glance

- Spread in both dollar and basis points
- Order book imbalance percentage
- VWAP and volume nodes
- Large order detection
- Trade momentum across 1m, 5m, 15m windows

### Visual Elements

- **Gauges**: Single and dual-value gauges for RSI, Stochastic
- **Depth Bars**: Proportional visualization of order sizes
- **Pressure Bars**: Buy/sell volume ratios
- **Price Ladder**: Current price context with key levels marked
- **Histogram Visualization**: MACD histogram with directional indicators

## Running the Dashboard

```bash
# From project root
python run_dashboard.py

# Or directly
python -m src.ui.dashboard
```

## Keyboard Shortcuts

- `q` - Quit the application
- `s` - Switch between symbols (BTC-USD, ETH-USD, SOL-USD)
- `d` - Toggle density mode (future feature)

## Architecture

The dashboard follows our Pydantic-first philosophy:

```python
# All data flows through Pydantic models
MarketSnapshot → TechnicalAnalysisService → BaseIndicator → Dashboard

# Reactive updates using Textual's reactive system
order_book: reactive[OrderBook | None] = reactive(None)
```

## Widget Components

### OrderBookDepthWidget

- Displays top 10 bid/ask levels with size visualization
- Calculates and shows spread metrics
- Shows order book imbalance (bid heavy/ask heavy/balanced)
- Color-coded by distance from mid price

### CompactIndicatorGrid

- Displays all indicators in a dense grid format
- Each indicator gets a mini-panel with:
  - Current value and state
  - Visual gauge or chart
  - Signal bias (bullish/bearish/neutral)
  - Special alerts (crossovers, divergences)

### MarketProfileWidget

- Shows price ladder with ±0.2% range
- Marks key levels:
  - VWAP
  - Volume nodes
  - Large bid/ask walls
- Updates in real-time with current price

### TradeMomentumWidget

- Analyzes trade flow in time buckets
- Shows buy/sell pressure bars
- Highlights large trades
- Indicates momentum direction

## Data Flow

1. **Market Data Input**: Real-time WebSocket or mock data
2. **Snapshot Creation**: Aggregates ticker, order book, and trades
3. **Technical Analysis**: Calculates indicators via TechnicalAnalysisService
4. **Widget Updates**: Reactive properties trigger re-renders
5. **Visual Display**: Rich text formatting with colors and symbols

## Customization

The dashboard is designed to be extended:

```python
# Add new indicators by implementing BaseIndicator
class MyIndicator(BaseIndicator):
    def suggest_signal(self) -> dict[str, Any]:
        return {"bias": "bullish", "strength": 0.8}

# Register with the IndicatorRegistry
registry.register("my_indicator", MyIndicator)
```

## Performance

- Efficient updates using Textual's reactive system
- Caching in TechnicalAnalysisService (2-second TTL)
- Minimal re-renders through targeted widget updates
- Mock data runs at 2 updates/second for smooth visualization

## Future Enhancements

- WebSocket integration for real market data
- Multi-exchange support
- Custom indicator configuration
- Trade execution integration
- Historical data playback
- Alert system for significant events
