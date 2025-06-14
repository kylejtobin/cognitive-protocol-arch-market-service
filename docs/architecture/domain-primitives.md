# Domain Primitives Architecture

## Overview

Domain primitives provide semantic meaning to values in our market data system. They work in harmony with our protocol-based architecture, where protocols define neutral contracts and domain models provide rich behavior.

## The Protocol-Model-Primitive Architecture

```mermaid
graph TB
    subgraph "External Layer"
        EXT1[Coinbase WebSocket]
        EXT2[Binance API]
        EXT3[Test Data]
    end

    subgraph "Adapter Layer"
        ADP1[CoinbaseTicker<br/>price_raw: str]
        ADP2[BinanceTicker<br/>price: float]
        ADP3[MockTicker<br/>price: Decimal]
    end

    subgraph "Protocol Layer - Neutral Contracts"
        PROTO[MarketTickerProtocol<br/>price: Decimal ← Neutral Type<br/>symbol: str ← Neutral Type]
    end

    subgraph "Domain Layer"
        DOM[MarketTicker<br/>_price: Price ← Domain Primitive<br/>symbol: str]
        PRIM[Domain Primitives<br/>Price, Size, Spread, etc.]
    end

    EXT1 --> ADP1
    EXT2 --> ADP2
    EXT3 --> ADP3

    ADP1 -->|Satisfies| PROTO
    ADP2 -->|Satisfies| PROTO
    ADP3 -->|Satisfies| PROTO
    DOM -->|Satisfies| PROTO

    DOM --> PRIM

    style PROTO fill:#ffffcc,stroke:#333,stroke-width:4px
    style PRIM fill:#ffccff,stroke:#333,stroke-width:2px
```

## Domain Primitives Design

### Core Primitives

```mermaid
classDiagram
    class Price {
        +value: Decimal
        +currency: str
        +to_decimal() Decimal
        +as_float() float
        +percentage_change(other) Percentage
        +with_spread(spread) Price
        +format_display() str
    }

    class Size {
        +value: Decimal
        +to_decimal() Decimal
        +as_float() float
        +is_zero() bool
        +format_display() str
    }

    class Percentage {
        +value: Decimal
        +as_decimal() Decimal
        +as_basis_points() BasisPoints
        +format_display() str
    }

    class BasisPoints {
        +value: Decimal
        +to_percentage() Percentage
        +as_decimal() Decimal
        +format_display() str
    }

    class Spread {
        +value: Decimal
        +reference_price: Price
        +as_percentage() Percentage
        +as_basis_points() BasisPoints
        +is_tight() bool
    }

    class Volume {
        +size: Size
        +timeframe_hours: int
        +daily_equivalent() Size
        +format_display() str
    }

    class VWAP {
        +price: Price
        +volume: Volume
        +confidence: Decimal
        +to_price() Price
    }

    Price --> Percentage : calculates
    Price --> Spread : creates
    Spread --> Percentage : converts to
    Spread --> BasisPoints : converts to
    Percentage --> BasisPoints : converts to
    Volume --> Size : contains
    VWAP --> Price : contains
    VWAP --> Volume : weighted by
```

## Protocol Satisfaction Pattern

The key insight is that protocols define **what** data is needed, while domain primitives define **how** to work with that data:

```mermaid
graph LR
    subgraph "Protocol Definition"
        P1[price: Decimal]
        P2[size: Decimal]
    end

    subgraph "Domain Model"
        D1[_price: Price]
        D2[_size: Size]
    end

    subgraph "Protocol Satisfaction"
        S1["@property<br/>def price(self) -> Decimal:<br/>    return self._price.value"]
        S2["@property<br/>def size(self) -> Decimal:<br/>    return self._size.value"]
    end

    D1 --> S1
    D2 --> S2
    S1 --> P1
    S2 --> P2

    style P1 fill:#ffffcc
    style P2 fill:#ffffcc
    style D1 fill:#ffccff
    style D2 fill:#ffccff
```

## Data Flow Example

Here's how data flows through the system with domain primitives:

```mermaid
sequenceDiagram
    participant API as External API
    participant Adapter as Adapter Layer
    participant Protocol as Protocol Layer
    participant Domain as Domain Model
    participant Primitive as Domain Primitive
    participant Analysis as Analysis Model

    API->>Adapter: {"price": "50000.50"}
    Adapter->>Protocol: price = Decimal("50000.50")
    Protocol->>Domain: Create domain model
    Domain->>Primitive: Price(value=Decimal("50000.50"))
    Domain->>Analysis: Pass model with primitives
    Analysis->>Analysis: spread.as_basis_points()
    Analysis->>Analysis: price.percentage_change()
    Note over Analysis: Rich operations on primitives
```

## Benefits

1. **Semantic Clarity**: `price.calculate_stop_loss(0.02)` vs `price * Decimal("0.98")`
2. **Encapsulated Logic**: VWAP calculation in one place
3. **Type Safety**: Can't accidentally pass a size where a price is expected
4. **Protocol Compliance**: Still satisfies all protocol contracts
5. **Testability**: Test primitive behavior once, use everywhere

## Example Usage

```python
# External adapter - no knowledge of primitives
class CoinbaseTicker(BaseModel):
    price_raw: str = Field(alias="price")

    @property
    def price(self) -> Decimal:  # Satisfies protocol
        return Decimal(self.price_raw)

# Domain model - rich primitives internally
class MarketTicker(BaseModel):
    _price: Price

    @property
    def price(self) -> Decimal:  # Satisfies protocol
        return self._price.value

    def calculate_spread_to(self, other: "MarketTicker") -> Spread:
        """Rich domain operation"""
        return Spread(
            value=abs(self._price.value - other._price.value),
            reference_price=self._price
        )

# Usage in analysis
def analyze_market(ticker: MarketTickerProtocol) -> dict:
    # Works with any implementation
    raw_price = ticker.price  # Always Decimal

    # Convert to domain model for rich operations
    if isinstance(ticker, MarketTicker):
        spread_bps = ticker._price.to_spread().as_basis_points()
    else:
        # Fallback for non-domain models
        spread_bps = calculate_spread_basis_points(raw_price)
```

## Migration Strategy

The beauty of this design is that it can be adopted incrementally:

1. **Phase 1**: Implement primitives
2. **Phase 2**: Update domain models to use primitives internally
3. **Phase 3**: Gradually migrate analysis code to use primitive methods
4. **Phase 4**: Update UI to use formatting methods

Throughout all phases, protocols remain unchanged and all existing code continues to work.
