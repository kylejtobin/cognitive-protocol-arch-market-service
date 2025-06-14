# Domain Primitives Implementation Plan

## Overview

This document outlines the plan to introduce domain primitives (Price, Size, Percentage, etc.) throughout the codebase while maintaining our protocol-based architecture with neutral types at boundaries.

## Guiding Principles

1. **Protocol Neutrality**: Protocols continue to use `Decimal` for price/size fields
2. **Domain Richness**: Domain models use primitives internally for semantic clarity
3. **Boundary Conversion**: Adapters convert between raw types and domain primitives
4. **Incremental Migration**: Use mypy --strict to guide the migration process
5. **Backward Compatibility**: Ensure protocols remain satisfied throughout migration

## Phase 1: Core Primitive Implementation

### 1.1 Create Base Domain Primitives

Location: `src/market/domain/primitives.py`

```python
# Core primitives to implement:
- Price: Financial price with currency context
- Size: Quantity/volume of an asset
- Percentage: Percentage values (50 = 50%)
- BasisPoints: Basis points (100 = 1%)
- Spread: Price spread with reference price
- Volume: Trading volume with time context
- VWAP: Volume-weighted average price
```

### 1.2 Key Features for Each Primitive

- Immutable (frozen=True)
- Validation (non-negative for prices, etc.)
- Conversion methods (to_decimal(), as_float())
- Rich comparison operators
- Arithmetic operations where sensible
- Formatting methods for display

### 1.3 Protocol Satisfaction

Each primitive must be able to satisfy protocol requirements:

```python
class Price(BaseModel):
    value: Decimal

    def to_decimal(self) -> Decimal:
        """For protocol satisfaction"""
        return self.value
```

## Phase 2: Update Domain Models

### 2.1 Market Models

Update in order of dependency:

1. **MarketTicker** (`src/market/model/ticker.py`)

   - Change `price: Decimal` to `_price: Price`
   - Add property to satisfy protocol
   - Update spread/mid_price calculations

2. **MarketTrade** (`src/market/model/trade.py`)

   - Change price and size fields
   - Maintain protocol compatibility

3. **OrderBook** (`src/market/model/book.py`)

   - Update price levels to use primitives
   - Enhance spread calculations

4. **MarketSnapshot** (`src/market/model/snapshot.py`)
   - No direct changes, but verify composition works

### 2.2 Analysis Models

Update analysis models to leverage primitives:

1. **SpreadAnalysis** (`src/market/analysis/models.py`)

   - Use Spread and BasisPoints types
   - Simplify calculations

2. **TradeFlowAnalysis**

   - Use VWAP type for vwap field
   - Use Volume for volume fields

3. **Technical Indicators**
   - Update each indicator to use appropriate primitives

## Phase 3: Update Adapters

### 3.1 Coinbase Adapter

- Keep raw string/float fields
- Convert to primitives in transformation methods
- Maintain protocol properties returning Decimal

### 3.2 Add Factory Methods

Create factory methods for easy conversion:

```python
@classmethod
def from_ticker_data(cls, data: TickerData) -> MarketTicker:
    return cls(
        _price=Price(value=data.price),  # Decimal from property
        ...
    )
```

## Phase 4: Update Service Layer

### 4.1 Technical Analysis Service

- Update to work with primitives
- Maintain backward compatibility for caching

### 4.2 Analysis Pipeline

- Leverage rich primitive methods
- Simplify transformation logic

## Phase 5: Update UI Layer

### 5.1 Dashboard Updates

- Use primitive formatting methods
- Replace manual calculations with primitive methods
- Update type hints throughout

## Phase 6: Test Updates

### 6.1 Test Helpers

- Update builders to work with primitives
- Add primitive factory methods

### 6.2 Unit Tests

- Update assertions to work with primitives
- Test primitive-specific behavior

### 6.3 Integration Tests

- Ensure end-to-end flows work
- Verify protocol satisfaction

## Implementation Strategy

### Step 1: Implement Core Primitives

1. Create `src/market/domain/` directory
2. Implement `primitives.py` with all domain primitives
3. Add comprehensive tests for primitives
4. Run `mypy --strict src/market/domain/`

### Step 2: Update One Model Chain

Start with ticker → trade → analysis flow:

1. Update MarketTicker to use Price internally
2. Run `mypy --strict` and fix type errors
3. Update dependent models
4. Update tests for that model

### Step 3: Cascade Updates

Use mypy errors to guide the cascade:

1. Run `mypy --strict src/`
2. Fix type errors module by module
3. Run tests after each module update
4. Commit working increments

### Step 4: Update UI and Helpers

1. Update UI to use formatting methods
2. Update test helpers
3. Final mypy strict check

## Validation Checklist

- [ ] All protocols still return Decimal for price/size
- [ ] All domain models use primitives internally
- [ ] All tests pass
- [ ] `mypy --strict src/` passes
- [ ] `ruff check src/` passes
- [ ] UI displays correctly
- [ ] Performance is not degraded

## Migration Tools

### Type Checking Commands

```bash
# Check specific module
mypy --strict src/market/model/ticker.py

# Check all with specific focus
mypy --strict src/ | grep -E "(Price|Size|Decimal)"

# Progressive checking
mypy --strict src/market/domain/
mypy --strict src/market/model/
mypy --strict src/market/analysis/
mypy --strict src/market/service/
mypy --strict src/
```

### Testing Commands

```bash
# Test primitives first
pytest tests/unit/market/test_primitives.py -xvs

# Test model by model
pytest tests/unit/market/test_ticker.py -xvs

# Run all tests
pytest -xvs
```

## Rollback Plan

If issues arise:

1. Domain primitives can coexist with Decimal
2. Can be rolled back module by module
3. Git branch protection ensures clean rollback

## Success Criteria

1. **Type Safety**: mypy --strict passes
2. **Semantic Clarity**: Code expresses domain concepts clearly
3. **No Regressions**: All tests pass
4. **Performance**: No degradation in performance
5. **Developer Experience**: Improved autocomplete and discoverability

## Estimated Timeline

- Phase 1 (Core Primitives): 2-3 hours
- Phase 2 (Domain Models): 3-4 hours
- Phase 3 (Adapters): 2 hours
- Phase 4 (Services): 2 hours
- Phase 5 (UI): 1 hour
- Phase 6 (Tests): 3-4 hours

Total: ~15-20 hours of focused work

## Next Steps

1. Review and refine this plan
2. Create the domain package structure
3. Implement Price primitive first
4. Use TDD to build out primitives
5. Begin incremental migration

---

_Note: This is a living document. Update as we discover new requirements or challenges during implementation._
