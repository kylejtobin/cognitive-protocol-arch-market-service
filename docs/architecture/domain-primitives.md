# Domain Primitives Architecture

## Overview

Domain primitives provide semantic meaning to values in our market data system. They work in harmony with our protocol-based architecture, where protocols define neutral contracts and domain models provide rich behavior through computed fields.

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
        DOM[MarketTicker<br/>price: Decimal ← Protocol Field<br/>@computed_field price_primitive: Price]
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

## The Computed Field Pattern

The key insight is using Pydantic's `@computed_field` to expose domain primitives while storing protocol-compliant types:

```python
class MarketTicker(BaseModel):
    # Store protocol-compliant types as fields
    price: Decimal  # Satisfies protocol directly
    size: Decimal
    
    # Expose domain primitives as computed fields
    @computed_field  # type: ignore[misc]
    @property
    def price_primitive(self) -> Price:
        """Price as domain primitive for calculations."""
        return Price(value=self.price)
    
    @computed_field  # type: ignore[misc]
    @property
    def size_primitive(self) -> Size:
        """Size as domain primitive for calculations."""
        return Size(value=self.size)
    
    # Rich domain operations use primitives internally
    def price_change_from(self, other: MarketTickerProtocol) -> Percentage:
        """Calculate percentage change - works with ANY protocol implementation!"""
        other_price = Price(value=other.price)
        return self.price_primitive.percentage_change(other_price)
```

### Benefits of Computed Fields

1. **No Storage Duplication**: Store data once, compute primitives on demand
2. **Cached by Default**: Pydantic caches computed fields for performance
3. **Clean Serialization**: Computed fields excluded from JSON by default
4. **Simple Mental Model**: Fields store data, computed fields derive state
5. **Type Safety**: Full type checking for both storage and primitives

## Domain Primitives Design

### Core Primitives

```mermaid
classDiagram
    class Price {
        +value: Decimal
        +currency: str = "USD"
        +to_decimal() Decimal
        +as_float() float
        +percentage_change(other) Percentage
        +calculate_stop_loss(percentage) Price
        +calculate_take_profit(percentage) Price
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
        +bid_price: Price
        +ask_price: Price
        +value() Decimal
        +mid_price() Price
        +as_percentage() Percentage
        +as_basis_points() BasisPoints
        +is_inverted() bool
        +is_crossed() bool
        +format_display() str
    }

    class Volume {
        +size: Size
        +timeframe_hours: int = 24
        +daily_equivalent() Size
        +format_display() str
    }

    class VWAP {
        +price: Price
        +volume: Volume
        +confidence: Decimal
        +to_decimal() Decimal
        +to_price() Price
        +deviation_from(price) Percentage
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

The computed field pattern provides the cleanest way to satisfy protocols while maintaining rich domain behavior:

```mermaid
graph LR
    subgraph "Protocol Definition"
        P1[price: Decimal]
        P2[size: Decimal]
    end

    subgraph "Domain Model Fields"
        D1[price: Decimal]
        D2[size: Decimal]
    end

    subgraph "Computed Primitives"
        C1["@computed_field<br/>price_primitive: Price"]
        C2["@computed_field<br/>size_primitive: Size"]
    end

    D1 --> P1
    D2 --> P2
    D1 --> C1
    D2 --> C2

    style P1 fill:#ffffcc
    style P2 fill:#ffffcc
    style C1 fill:#ffccff
    style C2 fill:#ffccff
```

## Data Flow Example

Here's how data flows through the system with domain primitives:

```mermaid
sequenceDiagram
    participant API as External API
    participant Adapter as Adapter Layer
    participant Protocol as Protocol Layer
    participant Domain as Domain Model
    participant Primitive as Computed Primitive
    participant Analysis as Analysis Service

    API->>Adapter: {"price": "50000.50"}
    Adapter->>Protocol: price = Decimal("50000.50")
    Protocol->>Domain: MarketTicker(price=...)
    Domain->>Domain: Store as Decimal field
    Analysis->>Domain: ticker.price_primitive
    Domain->>Primitive: @computed_field creates Price
    Primitive->>Analysis: Price instance
    Analysis->>Analysis: price.percentage_change()
    Note over Analysis: Rich operations on primitives
```

## Real Implementation Example

```python
class MarketTicker(BaseModel):
    """Market ticker with rich domain behavior."""
    
    # Protocol-compliant fields (stored as Decimal)
    symbol: str = Field(..., min_length=1, max_length=20)
    price: Decimal = Field(..., gt=0)
    size: Decimal = Field(..., gt=0)
    bid: Optional[Decimal] = Field(None, gt=0)
    ask: Optional[Decimal] = Field(None, gt=0)
    
    # Computed domain primitives
    @computed_field  # type: ignore[misc]
    @property
    def price_primitive(self) -> Price:
        return Price(value=self.price)
    
    @computed_field  # type: ignore[misc]
    @property
    def spread(self) -> Optional[Decimal]:
        """Computed spread value."""
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid
    
    # Rich domain operations
    def spread_percentage(self) -> Optional[Percentage]:
        """Calculate spread as percentage of mid price."""
        if self.spread is None:
            return None
        mid = (self.bid + self.ask) / Decimal("2")
        return Percentage(value=(self.spread / mid) * 100)
    
    def is_liquid(self, max_spread_bps: int = 10) -> bool:
        """Check if market is liquid."""
        spread_bps = self.spread_basis_points()
        return spread_bps is not None and spread_bps <= max_spread_bps
```

## Benefits

1. **Semantic Clarity**: `ticker.price_primitive.calculate_stop_loss(Decimal("0.02"))` expresses intent
2. **No Storage Overhead**: Primitives computed only when accessed
3. **Protocol Compliance**: Models satisfy protocols without modification
4. **Type Safety**: Can't mix prices and sizes
5. **Clean Serialization**: `model_dump()` excludes computed fields by default

## Anti-Patterns to Avoid

```python
# ❌ Don't use PrivateAttr
class BadModel(BaseModel):
    price_input: Decimal = Field(alias="price")
    _price: Price = PrivateAttr()  # Complex and confusing

# ❌ Don't store computed values
class BadModel2(BaseModel):
    price: Decimal
    price_doubled: Decimal  # Should be @computed_field

# ❌ Don't use regular @property
class BadModel3(BaseModel):
    @property
    def spread(self) -> Decimal:  # Use @computed_field!
        return self.ask - self.bid
```

## Migration Strategy

The computed field pattern makes migration straightforward:

1. **Phase 1**: Add computed primitive properties to existing models
2. **Phase 2**: Update domain operations to use primitives internally  
3. **Phase 3**: Services can gradually adopt primitive usage
4. **Phase 4**: UI can use primitive formatting methods

Throughout all phases, protocols remain unchanged and all existing code continues to work. The pattern supports incremental adoption without breaking changes.
