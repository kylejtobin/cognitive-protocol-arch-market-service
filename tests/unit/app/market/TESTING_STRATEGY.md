# Market Service Testing Strategy

## Philosophy

Our tests should:

- **Verify behavior, not implementation** - Test what the system does, not how
- **Be maintainable** - Clear, focused, and easy to update
- **Provide fast feedback** - Most tests run in milliseconds
- **Document the system** - Tests as living documentation
- **Catch real bugs** - Focus on likely failure modes

## Test Pyramid

### 1. Unit Tests (70% of tests)

Fast, isolated tests of individual components.

#### What to Test:

- **Protocol Satisfaction**: Verify our models satisfy protocols structurally
- **Data Parsing**: Pydantic models correctly parse various JSON formats
- **Business Logic**: Order book calculations, trade aggregation
- **Edge Cases**: Empty data, None values, extreme numbers

#### What NOT to Test:

- Simple getters/setters
- Pydantic's validation (trust the framework)
- Python's built-in behavior

#### Example Focus Areas:

```python
# ✅ Good: Test business logic
def test_order_book_spread_calculation():
    """Spread should be ask - bid"""

# ❌ Bad: Test framework behavior
def test_pydantic_validates_decimal():
    """Don't test what Pydantic already guarantees"""
```

### 2. Integration Tests (20% of tests)

Test component interactions without external dependencies.

#### What to Test:

- **Message Flow**: WebSocket message → Parser → State update → Snapshot
- **State Management**: Order book updates maintain consistency
- **Error Handling**: Malformed messages don't crash the system
- **Concurrency**: Multiple symbols update independently

#### Techniques:

- Mock WebSocket client with recorded messages
- Use test fixtures from actual Coinbase data
- Time-based tests with controlled clocks

### 3. End-to-End Tests (10% of tests)

Test with real external systems (run separately).

#### What to Test:

- **Connection**: Can connect to Coinbase WebSocket
- **Subscription**: Can subscribe to channels
- **Data Flow**: Receive and parse real messages
- **Reconnection**: Handle disconnects gracefully

#### Considerations:

- Run in separate test suite (not in CI)
- Use Coinbase sandbox when available
- Rate limit aware
- Network failure tolerant

## Testing Patterns

### 1. Fixture-Based Testing

Use real Coinbase message samples as fixtures:

```python
@pytest.fixture
def ticker_message():
    """Real ticker message from Coinbase."""
    return load_fixture("coinbase/ticker/btc_usd_sample.json")
```

### 2. Builder Pattern for Test Data

Make test data creation easy and clear:

```python
def order_book_builder() -> OrderBookBuilder:
    return (OrderBookBuilder()
        .with_symbol("BTC-USD")
        .with_bid(45000, size=1.0)
        .with_ask(45100, size=1.0))
```

### 3. Parameterized Tests

Test multiple scenarios efficiently:

```python
@pytest.mark.parametrize("price,expected", [
    ("100.50", Decimal("100.50")),
    (100.50, Decimal("100.50")),
    ("0", Decimal("0")),
])
def test_price_parsing(price, expected):
    """Test various price formats."""
```

### 4. Mock Strategies

#### For WebSocket Client:

```python
class MockWSClient:
    """Controllable WebSocket for testing."""

    def emit_message(self, channel: str, data: dict):
        """Simulate incoming message."""

    def simulate_disconnect(self):
        """Test reconnection logic."""
```

#### For Time:

```python
@freeze_time("2024-01-01 10:00:00")
def test_snapshot_timestamp():
    """Test with controlled time."""
```

## Critical Test Scenarios

### 1. Data Integrity

- Decimal precision is maintained
- Timestamps are properly converted
- Symbol mapping is consistent

### 2. State Consistency

- Order book updates are atomic
- Sequence gaps are detected
- Snapshots reflect current state

### 3. Error Resilience

- Malformed JSON is handled
- Missing fields use defaults
- Type mismatches are logged

### 4. Performance Boundaries

- Large order books (1000+ levels)
- High message rates (100+ msg/sec)
- Memory usage stays bounded

## Anti-Patterns to Avoid

### 1. Testing Internals

```python
# ❌ Bad: Tests implementation details
def test_ticker_has_raw_field():
    assert hasattr(ticker, 'price_raw')

# ✅ Good: Tests behavior
def test_ticker_provides_decimal_price():
    assert isinstance(ticker.price, Decimal)
```

### 2. Overly Coupled Tests

```python
# ❌ Bad: One failure cascades
def test_everything():
    # Connect, subscribe, parse, calculate...

# ✅ Good: Focused tests
def test_spread_calculation():
    # Just test spread logic
```

### 3. Magic Numbers

```python
# ❌ Bad: What is 0.01106?
assert spread_pct == 0.01106

# ✅ Good: Clear calculation
expected = (spread / mid_price) * 100
assert spread_pct == pytest.approx(expected)
```

## Test Organization

```
tests/
├── unit/app/market/
│   ├── test_structural_typing.py   # Protocol satisfaction
│   ├── test_data_parsing.py        # JSON → Model
│   ├── test_order_book.py          # Business logic
│   └── test_error_handling.py      # Edge cases
├── integration/app/market/
│   ├── test_message_flow.py        # Full processing
│   ├── test_state_management.py    # Multi-symbol
│   └── test_mock_websocket.py      # Connection scenarios
└── e2e/ (separate)
    └── test_live_coinbase.py        # Real connection
```

## Continuous Improvement

1. **Track Test Metrics**:

   - Coverage (aim for 80%+ on critical paths)
   - Execution time (flag slow tests)
   - Flakiness (eliminate non-deterministic tests)

2. **Review Test Value**:

   - Has this test caught a real bug?
   - Does it test something important?
   - Is it maintainable?

3. **Refactor Tests**:
   - Extract common patterns
   - Improve test names
   - Update for API changes

## Next Steps

1. Implement MockWSClient for controlled testing
2. Create fixture loader for Coinbase messages
3. Build test data builders for common scenarios
4. Set up separate test commands:
   - `pytest tests/unit` - Fast feedback
   - `pytest tests/integration` - Component integration
   - `pytest tests/e2e --coinbase-live` - Real connection
