# Cognitive Protocol Architecture Market Service

A **production-grade** demonstration of **[Cognitive Protocol Architecture (CPA)](docs/philosophy.md)** - a design pattern that unifies domain modeling, type safety, and AI readiness into a single coherent approach. Built with Python 3.13+, this real-time market data service shows how CPA reduces code by 50-70% while making systems naturally interpretable by both humans and AI.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/kylejtobin/cognitive-protocol-arch-market-service.git
cd cognitive-protocol-arch-market-service

# Install dependencies (including dev tools)
uv sync --extra dev

# Set your Coinbase API credentials
export COINBASE_API_KEY="your-api-key"
export COINBASE_API_SECRET="your-api-secret"

# Run the dashboard
python run_dashboard.py
```

## Live Dashboard

![Market Service Dashboard](img/ui.png)

The terminal dashboard displays real-time:
- **Order book visualization** with bid/ask depth
- **Trade flow** showing buy/sell pressure
- **Technical indicators** (RSI, MACD, Stochastic, Volume)
- **Price levels** with support/resistance zones

Built with [Textual](https://github.com/Textualize/textual) for a responsive terminal UI.

## What This Demonstrates

This service showcases CPA principles in action:

- **Real-time WebSocket processing** from Coinbase
- **Zero defensive programming** - models validate themselves
- **Natural data flow** - no mappers or DTOs needed
- **Technical analysis pipeline** that's ML-ready
- **3,000 lines of code** doing what traditionally takes 10,000+

## Project Structure

```
src/market/
├── protocols/      # Interface contracts (what we need)
├── models/         # Domain models (Pydantic + behavior)
├── adapters/       # External API integration
├── analysis/       # Technical indicators (RSI, MACD, etc.)
├── service/        # Business orchestration
├── connection/     # WebSocket management
└── ui/             # Terminal dashboard components
```

## Key Features

### Real-Time Market Data
- Connects to Coinbase WebSocket API
- Processes order book updates, trades, and tickers
- Maintains synchronized market state

### Technical Analysis
- Rolling window calculations for price series
- Momentum indicators (RSI, MACD, Stochastic)
- Volume analysis with buy/sell pressure
- Multi-timeframe support

### Architecture Benefits
- **Type-safe throughout** with `mypy --strict`
- **Async-first** using Python 3.13+ features
- **Self-validating models** via Pydantic v2
- **Natural composition** - models transform into each other

## Example: Adding a New Exchange

With CPA, integrating a new data source is simple:

```python
# 1. Create an adapter model for the exchange's format
class KrakenTicker(BaseModel):
    pair: str = Field(alias="wsname")
    last: Decimal = Field(alias="c[0]")
    
    def to_market_ticker(self) -> MarketTicker:
        return MarketTicker(
            symbol=self.pair,
            price=self.last,
            timestamp=datetime.now(UTC)
        )

# 2. Use it - that's all!
ticker = KrakenTicker.model_validate(kraken_data)
market_ticker = ticker.to_market_ticker()
```

No need to create DTOs, mappers, validators, or update dependency injection.

## Development

### Running Tests

```bash
# Run all tests with coverage
pytest --cov=market --cov-report=term-missing

# Type checking
mypy src/ --strict

# Code formatting
ruff check src/
black src/ --check
```

### Dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable dependency management:

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync --extra dev
```

## Technical Stack

- **Python 3.13+** with asyncio
- **Pydantic v2** for validation and serialization
- **Pandas** for time series analysis
- **Textual** for the terminal UI
- **WebSocket** for real-time data
- **uv** for package management

## Learn More

- **CPA Deep Dive**: [Understanding Cognitive Protocol Architecture](docs/philosophy.md)
- **Blog Post**: [Introducing CPA - When the Model Becomes the Contract](#)
- **Discussion**: [HN Thread](#) | [Reddit](#)

## License

MIT

---

*This project demonstrates how thoughtful architecture can dramatically simplify complex systems while making them naturally ready for AI integration.*
