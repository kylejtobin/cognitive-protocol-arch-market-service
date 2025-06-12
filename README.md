# Protocol Architecture Market Service

A **production-grade**, async Python market data service demonstrating how Protocol-based architecture creates cleaner, more maintainable systems than traditional hexagonal patterns. Built with Python 3.13+, full type safety, and designed for real-time financial data processing.

## Key Features

- 🚀 **Async-first** with WebSocket streaming and concurrent indicator calculations
- 🔒 **Type-safe** with Protocols, Pydantic v2, and mypy strict mode
- 📊 **Real-time analysis** of market microstructure and momentum indicators
- 🧠 **AI/ML ready** with semantic protocols and multi-level data representation
- 📈 **High-performance** using Pandas vectorization and caching strategies
- 🎨 **Beautiful TUI** with real-time order book visualization

## Why Protocol Architecture?

Traditional hexagonal architecture requires extensive boilerplate:

- Separate DTOs, domain models, and mappers
- Explicit ports and adapters interfaces
- Defensive programming throughout
- 3-5 classes for each concept

Protocol Architecture leverages Python's strengths:

- **Structural typing via Protocols** - Define what you need, not how it's implemented
- **Pydantic models as domain objects** - Validation, serialization, and business logic in one place
- **Composition over layers** - Models flow into each other naturally
- **50% less code** - Same safety, better ergonomics

## Quick Start

```bash
# Install dependencies with uv
uv sync

# Set your Coinbase API credentials
export COINBASE_API_KEY="your-api-key"
export COINBASE_API_SECRET="your-api-secret"

# Run the terminal dashboard
python run_dashboard.py
```

## Live Dashboard

![Market Service Dashboard](img/ui.png)

The terminal dashboard showcases real-time market analysis with:

- **Order book depth visualization** with liquidity heatmaps
- **Trade momentum indicators** showing buy/sell pressure
- **Technical indicators grid** with RSI, MACD, Stochastic, and Volume analysis
- **Price level mapping** highlighting key support/resistance zones

Built with [Textual](https://github.com/Textualize/textual) for a responsive TUI that updates in real-time.

## Architecture Overview

### 1. Protocols Define Contracts (`src/market/protocols/`)

```python
@runtime_checkable
class MarketTickerProtocol(Protocol):
    """What a ticker IS, not how it's implemented."""
    @property
    def symbol(self) -> str: ...
    @property
    def price(self) -> Decimal: ...
    @property
    def timestamp(self) -> datetime: ...
```

### 2. Pydantic Models Implement Protocols (`src/market/models/`)

```python
class MarketTicker(BaseModel):
    """A model that automatically satisfies MarketTickerProtocol."""
    symbol: str
    price: Decimal
    timestamp: datetime

    # Validation built-in
    @field_validator("price")
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("Price must be positive")
        return v
```

### 3. Adapters Transform External Data (`src/market/adapters/`)

```python
class CoinbaseTicker(BaseModel):
    """External API shape with automatic transformation."""
    price: Decimal = Field(alias="price")
    product_id: str = Field(alias="product_id")

    def to_market_ticker(self) -> MarketTicker:
        """Transform to domain model."""
        return MarketTicker(
            symbol=self.product_id,
            price=self.price,
            timestamp=datetime.now(UTC)
        )
```

### 4. Services Compose Models (`src/market/service/`)

```python
# Models flow naturally without mappers
websocket_data → CoinbaseTicker → MarketTicker → MarketSnapshot → Analysis
```

## Key Concepts

### Protocol-Model Duality

Any Pydantic model that has the right fields automatically satisfies a protocol:

```python
# Protocol defines the contract
class TickerProtocol(Protocol):
    @property
    def price(self) -> Decimal: ...

# Multiple models can satisfy it
class CoinbaseTicker(BaseModel):
    price: Decimal  # ✓ Satisfies protocol

class BinanceTicker(BaseModel):
    price: Decimal  # ✓ Also satisfies protocol

# Use interchangeably
def process(ticker: TickerProtocol) -> None:
    print(f"Price: {ticker.price}")
```

### Composition as Architecture

Instead of layers, we have models that transform into each other:

```python
class SpreadAnalysis(BaseModel):
    """Analyzes bid/ask spreads."""

    @classmethod
    def from_snapshot(cls, snapshot: MarketSnapshot) -> "SpreadAnalysis":
        """Create from market snapshot."""
        return cls(
            spread=snapshot.calculate_spread(),
            spread_percentage=snapshot.spread_percentage
        )

    def to_signal(self) -> TradingSignal:
        """Transform to trading signal."""
        return TradingSignal(
            action="buy" if self.is_tight else "hold",
            confidence=self.calculate_confidence()
        )
```

### Fail-Fast Boundaries

Pydantic validates at the edge, eliminating defensive programming:

```python
# At the WebSocket boundary
data = websocket.receive()
ticker = CoinbaseTicker.model_validate_json(data)  # Fails here if invalid

# Rest of the system KNOWS data is valid
# No need for null checks or try/except blocks
```

## Project Structure

```
protocol-arch-market-svc/
├── src/market/
│   ├── protocols/       # Protocol definitions (contracts)
│   ├── models/          # Pydantic domain models
│   ├── adapters/        # External API adapters
│   ├── analysis/        # Market analysis components
│   ├── service/         # Service layer
│   ├── connection/      # WebSocket management
│   └── ui/             # Terminal dashboard
├── tests/              # Unit tests
└── docs/               # Architecture documentation
```

## Engineering Excellence

### Performance & Scalability

- **Async-first architecture** using Python 3.11+ with full `asyncio` support
- **Connection pooling** for WebSocket streams with automatic reconnection
- **In-memory caching** with TTL for expensive calculations
- **Pandas-optimized** technical analysis using vectorized operations
- **Zero-copy data flow** where possible using Pydantic's validation

### Type Safety & Developer Experience

```python
# Full type coverage with Python 3.11+ features
async def analyze_market(
    snapshot: MarketSnapshot,
    indicators: list[type[BaseIndicator]] | None = None,
) -> IndicatorResult:
    """Every function is fully typed with modern Python syntax."""
```

- **100% type coverage** validated with `mypy --strict`
- **Runtime validation** via Pydantic with detailed error messages
- **Protocol verification** using `@runtime_checkable` decorators
- **Self-documenting code** with semantic docstrings

### Testing & Quality

```bash
# Comprehensive test suite
pytest --cov=market --cov-report=term-missing

# Type checking
mypy src/ --strict

# Code quality
ruff check src/
black src/ --check
```

- **Property-based testing** for mathematical indicators
- **Async test fixtures** for WebSocket simulation
- **Snapshot testing** for UI components
- **CI/CD ready** with GitHub Actions workflows

### Observability & Debugging

The architecture provides multiple observation points:

```python
# Rich logging with structured data
logger.info("Market snapshot processed", extra={
    "symbol": snapshot.symbol,
    "latency_ms": processing_time,
    "indicators_calculated": len(result.indicators)
})

# Built-in performance metrics
@measure_performance
async def calculate_indicators(...) -> IndicatorResult:
    # Automatic latency tracking
```

## Running Tests

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
pytest

# With coverage
pytest --cov=market
```

## Design Benefits

1. **Type Safety Without Ceremony**

   - Protocols provide contracts
   - Pydantic enforces them
   - No manual validation code

2. **Testability**

   - Mock any protocol easily
   - Pydantic models are just data
   - No complex dependency injection

3. **Maintainability**

   - Change a field? Pydantic catches all usages
   - Add validation? One place, automatically applied
   - Refactor safely with type checking

4. **Performance**
   - No redundant transformations
   - Pydantic's Rust core is fast
   - Direct model usage, no mapping overhead

## Technical Analysis Service & Momentum Engine

_Why this part of the repo matters for AI / ML_

`src/market/service/technical_analysis.py` orchestrates a **chain-of-models** that turns raw snapshots into high-signal features:

```python
async def calculate_indicators(symbol: str, snapshot: MarketSnapshot) -> IndicatorResult:
    price_series = self.series_cache[symbol]

    rsi = RSIAnalysis.from_price_series(price_series)
    macd = MACDAnalysis.from_price_series(price_series)
    stochastic = StochasticAnalysis.from_price_series(price_series)
    volume = VolumeAnalysis.from_snapshot(snapshot)

    return IndicatorResult(
        indicators=[rsi, macd, stochastic, volume]
    )
```

Key observations:

1. **Protocol inputs, Pydantic outputs** – Each analysis class both implements and consumes protocols, so you get compile-time guarantees and runtime validation on every hop.
2. **Feature vector for free** – Need an ML feature store? `indicator.model_dump()` already gives you a JSON-serialisable, schema-validated record.
3. **Momentum focus** – The service keeps rolling windows of price/trade data, producing _momentum-oriented_ metrics (RSI, MACD histogram, stochastic K/D, buy-sell imbalance). These are exactly the signals most trading ML models start with.
4. **Composable pipeline** – Because every step is just a Pydantic model, you can slot in new indicators (or an ML inference result) without touching the surrounding plumbing.

> In other words, the market data service **is already a feature-engineering pipeline**. Hook a model on the end and you have a complete AI loop.

### Multi-Level Data Representation

Each indicator provides data at multiple abstraction levels - exactly what AI systems need:

```python
# RSIAnalysis provides:
rsi_value: 72.5                    # Raw numeric for ML features
momentum_state: "bullish"          # Categorical for classification
is_overbought: True               # Boolean for rule engines
semantic_summary(): "Bullish..."   # Natural language for LLMs

# Built-in methods for AI consumption:
.to_agent_context()  # Structured data for agents
.suggest_signal()    # Trading recommendations
.model_dump()        # Complete state for feature stores
```

The semantic protocols in `src/market/protocols/market.py` act as a **knowledge graph** that AI systems can traverse:

- Each protocol documents its "Semantic Role" and "Relationships"
- Docstrings are written to be both human AND machine-readable
- The type system itself encodes domain knowledge

This isn't just clean code - it's a **cognitive-ready architecture** where the codebase itself becomes training data for AI systems.

## Example: Adding a New Exchange

Traditional architecture: Create port, adapter, DTO, mapper, tests for each.

Protocol Architecture:

```python
# 1. Create adapter model
class BinanceTicker(BaseModel):
    s: str = Field(alias="symbol")
    c: Decimal = Field(alias="close")

    def to_market_ticker(self) -> MarketTicker:
        return MarketTicker(
            symbol=self.s,
            price=self.c,
            timestamp=datetime.now(UTC)
        )

# 2. Use it - automatically satisfies protocols
ticker = BinanceTicker.model_validate(binance_data)
market_ticker = ticker.to_market_ticker()
# Done! No other files to modify
```

## Development Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable Python package management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/kylefoley76/protocol-arch-market-svc.git
cd protocol-arch-market-svc
uv sync --extra dev

# Run type checking
uv run mypy src/ --strict

# Run linting
uv run ruff check src/
```

## Philosophy

> "Make the right thing easy and the wrong thing hard."

Protocol Architecture achieves this by:

- Making validation automatic (Pydantic)
- Making contracts implicit (Protocols)
- Making composition natural (method chaining)
- Making errors fail fast (at boundaries)

The result is a system that's both flexible and safe, with significantly less code than traditional approaches.

## License

MIT
