# Domain Primitives Implementation Plan

## Overview

This document outlines the plan to introduce domain primitives (Price, Size, Percentage, etc.) throughout the codebase while maintaining our protocol-based architecture with neutral types at boundaries.

## ✅ Key Discovery: Computed Field Pattern

We discovered that using Pydantic's `@computed_field` decorator provides the cleanest solution:

- Store protocol-compliant types as regular fields
- Expose domain primitives as computed properties
- No PrivateAttr complexity or duplicate storage
- Clean serialization and caching built-in

## Guiding Principles

1. **Protocol Neutrality**: Protocols continue to use `Decimal` for price/size fields
2. **Domain Richness**: Domain models expose primitives via computed fields
3. **Computed Field Pattern**: Use `@computed_field` for all derived state
4. **Incremental Migration**: Use mypy --strict to guide the migration process
5. **Backward Compatibility**: Ensure protocols remain satisfied throughout migration

## Phase 1: Core Primitive Implementation ✅ COMPLETED

### 1.1 Create Base Domain Primitives ✅

Location: `src/market/domain/primitives.py`

Implemented primitives:

- ✅ Price: Financial price with currency context
- ✅ Size: Quantity/volume of an asset
- ✅ Percentage: Percentage values (50 = 50%)
- ✅ BasisPoints: Basis points (100 = 1%)
- ✅ Spread: Price spread with bid/ask prices
- ✅ Volume: Trading volume with time context
- ✅ VWAP: Volume-weighted average price

### 1.2 Key Features for Each Primitive ✅

All primitives implemented with:

- ✅ Immutable (frozen=True)
- ✅ Validation (non-negative for prices, etc.)
- ✅ Conversion methods (to_decimal(), as_float())
- ✅ Rich comparison operators
- ✅ Arithmetic operations where sensible
- ✅ Formatting methods for display

### 1.3 Protocol Satisfaction ✅

Each primitive provides `to_decimal()` for protocol compatibility.

## Phase 2: Update Domain Models

### 2.1 Market Models

Update in order of dependency:

1. **MarketTicker** (`src/market/model/ticker.py`) ✅ COMPLETED

   - ✅ Store fields as Decimal (protocol-compliant)
   - ✅ Add computed fields for primitives (`price_primitive`, etc.)
   - ✅ Update spread/mid_price as computed fields
   - ✅ Rich domain methods use primitives internally

2. **MarketTrade** (`src/market/model/trade.py`) ⏳ TODO

   - Apply same computed field pattern
   - Maintain protocol compatibility

3. **OrderBook** (`src/market/model/book.py`) ⏳ TODO

   - Update price levels to expose primitives
   - Enhance spread calculations

4. **MarketSnapshot** (`src/market/model/snapshot.py`) ⏳ TODO
   - Verify composition works with updated models

### 2.2 Analysis Models ⏳ TODO

Update analysis models to leverage primitives:

1. **SpreadAnalysis** (`src/market/analysis/models.py`)

   - Use computed fields for Spread and BasisPoints
   - Simplify calculations

2. **TradeFlowAnalysis**

   - Use computed VWAP fields
   - Use Volume computed fields

3. **Technical Indicators**
   - Update each indicator to use appropriate primitives

## Phase 2.5: Analysis Component Primitives 🆕 NEXT

### Overview

Create domain primitives for technical indicator values to provide type safety and rich behavior for analysis components.

### Analysis Primitives to Implement

Location: `src/market/domain/analysis_primitives.py`

1. **RSIValue**

   ```python
   - value: Decimal (0-100)
   - zone: Literal["oversold", "neutral", "overbought"]
   - distance_from_neutral() -> Decimal
   - is_diverging(price_trend: Trend) -> bool
   ```

2. **MACDValue**

   ```python
   - macd_line: Decimal
   - signal_line: Decimal
   - histogram: Decimal
   - trend: Literal["bullish", "bearish", "neutral"]
   - crossover_strength() -> Percentage
   ```

3. **StochasticValue**

   ```python
   - k_value: Decimal (0-100)
   - d_value: Decimal (0-100)
   - zone: Literal["oversold", "neutral", "overbought"]
   - momentum: Literal["increasing", "decreasing", "stable"]
   ```

4. **BollingerBandPosition**

   ```python
   - price: Price
   - upper_band: Price
   - middle_band: Price
   - lower_band: Price
   - position_pct: Percentage  # Position within bands
   - band_width: Decimal
   - squeeze_detected: bool
   ```

5. **VolumeProfile**
   ```python
   - current_volume: Volume
   - average_volume: Volume
   - volume_ratio: Decimal
   - buy_pressure: Percentage
   - sell_pressure: Percentage
   - is_abnormal() -> bool
   ```

### Update Analysis Models

1. **RSIAnalysis** (`src/market/analysis/momentum_indicators.py`)

   - Add `rsi_value: Decimal` field
   - Add `@computed_field rsi_primitive` returning `RSIValue`
   - Update methods to use primitive

2. **MACDAnalysis** (`src/market/analysis/macd.py`)

   - Store raw values as Decimal
   - Expose MACD primitive via computed field
   - Simplify crossover detection

3. **StochasticAnalysis** (`src/market/analysis/stochastic.py`)

   - Similar pattern with StochasticValue

4. **VolumeAnalysis** (`src/market/analysis/volume_analysis.py`)
   - Use Volume primitives throughout
   - Add VolumeProfile computed field

### Benefits

- **Type Safety**: Can't have RSI > 100 or < 0
- **Rich Semantics**: `rsi.is_oversold` vs `rsi < 30`
- **Consistent Calculations**: Zone detection in one place
- **Better Testing**: Test primitives independently

## Phase 3: Update Adapters ⏳ TODO

### 3.1 Coinbase Adapter

- Keep raw string/float fields
- Protocol properties already return Decimal
- Domain models will handle primitive conversion

### 3.2 Factory Methods

No longer needed - Pydantic handles conversion automatically.

## Phase 4: Update Service Layer ⏳ TODO

### 4.1 Technical Analysis Service

- Update to work with primitives where beneficial
- Maintain backward compatibility for caching

### 4.2 Analysis Pipeline

- Leverage rich primitive methods
- Simplify transformation logic

## Phase 5: Update UI Layer ⏳ TODO

### 5.1 Dashboard Updates

- Use primitive formatting methods
- Replace manual calculations with primitive methods
- Update type hints throughout

## Phase 6: Test Updates

### 6.1 Test Helpers ⏳ TODO

- Update builders to work with models using computed fields
- Tests can continue using Decimal inputs

### 6.2 Unit Tests

- ✅ Primitive tests complete (29 tests passing)
- ✅ MarketTicker tests updated (13 tests passing)
- ⏳ Other model tests need updates

### 6.3 Integration Tests ⏳ TODO

- Ensure end-to-end flows work
- Verify protocol satisfaction

## Implementation Strategy

### ✅ Step 1: Implement Core Primitives - COMPLETED

1. ✅ Created `src/market/domain/` directory
2. ✅ Implemented `primitives.py` with all domain primitives
3. ✅ Added comprehensive tests for primitives
4. ✅ Validated with mypy

### Step 2: Update One Model Chain - IN PROGRESS

✅ MarketTicker completed with computed field pattern
⏳ Continue with trade → analysis flow

### Step 2.5: Analysis Primitives - NEXT

1. Create `analysis_primitives.py`
2. Implement RSIValue first (simplest)
3. Update RSIAnalysis to use it
4. Run tests and fix issues
5. Continue with other indicators

### Step 3: Cascade Updates ⏳ TODO

Use mypy errors to guide the cascade:

1. Run `mypy --strict src/`
2. Fix type errors module by module
3. Run tests after each module update
4. Commit working increments

### Step 4: Update UI and Helpers ⏳ TODO

1. Update UI to use formatting methods
2. Update test helpers
3. Final mypy strict check

## Validation Checklist

- ✅ All protocols still return Decimal for price/size
- ✅ Domain models use computed fields for primitives
- ✅ MarketTicker tests pass
- ⏳ `mypy --strict src/` passes
- ⏳ `ruff check src/` passes
- ⏳ UI displays correctly
- ⏳ Performance is not degraded

## Key Learnings

1. **Computed Field Pattern**: The `@computed_field` decorator is the key to clean architecture
2. **Type Ignore Needed**: Use `# type: ignore[misc]` for decorator order issues
3. **Serialization**: Computed fields are excluded by default - override `model_dump()` if needed
4. **Property Access**: Computed fields work like regular properties after creation
5. **Services ≠ Models**: Services orchestrate behavior, models hold data - don't mix them!

## Success Criteria

1. **Type Safety**: mypy --strict passes
2. **Semantic Clarity**: Code expresses domain concepts clearly
3. **No Regressions**: All tests pass
4. **Performance**: No degradation (computed fields are cached)
5. **Developer Experience**: Improved autocomplete and discoverability

## Progress Timeline

- ✅ Phase 1 (Core Primitives): Completed in ~2 hours
- ✅ Phase 2 (Domain Models): MarketTicker done, ~3 hours remaining
- ✅ Phase 2.5 (Analysis Primitives): Completed in ~1 hour
- ✅ RSIAnalysis updated to use RSIValue primitive
- ✅ All mypy --strict issues resolved
- ⏳ Phase 3 (Adapters): 2 hours estimated
- ⏳ Phase 4 (Services): 2 hours estimated
- ⏳ Phase 5 (UI): 1 hour estimated
- ✅ Phase 6 (Tests): RSI tests updated, ~1 hour remaining

Total Progress: ~40% complete (6 hours done, ~9 hours remaining)

## Next Steps

1. ✅ Price primitive implemented and tested
2. ✅ MarketTicker updated with computed field pattern
3. ✅ Create analysis primitives starting with RSIValue
4. ✅ Update RSIAnalysis to use RSIValue primitive
5. ⏳ Continue with other analysis components (MACD, Stochastic, etc.)
6. ⏳ Update MarketTrade model with primitives
7. ⏳ Update OrderBook model with primitives

---

_Note: This is a living document. Last updated after implementing analysis primitives and fixing all mypy issues._
