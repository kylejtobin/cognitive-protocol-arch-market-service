# Typed Agent Contexts in CPA

## Overview

While Cognitive Protocol Architecture (CPA) emphasizes simplicity and reduced code, maintaining type safety throughout the system provides significant benefits:

- **IDE Support**: Full IntelliSense and autocomplete
- **Compile-time Safety**: Catch errors before runtime
- **Self-documenting**: Types serve as documentation
- **AI-friendly**: Structured data is easier for LLMs to process

## Implementation

Instead of returning untyped `dict[str, Any]` from indicators, we use typed Pydantic models:

```python
# Before: Untyped dictionary
def to_agent_context(self) -> dict[str, Any]:
    return {
        "rsi": self.rsi_value,
        "state": self.momentum_state,
        "overbought": self.rsi_value > 70
    }

# After: Typed Pydantic model
def to_agent_context(self) -> RSIAgentContext:
    return RSIAgentContext(
        indicator="rsi",
        value=self.rsi_value,
        state=self.momentum_state,
        strength=self._calculate_strength(),
        key_levels=RSIKeyLevels(
            overbought=70,
            oversold=30,
            current=self.rsi_value
        ),
        signals=RSISignals(
            is_overbought=self.is_overbought,
            is_oversold=self.is_oversold,
            divergence=self.divergence_type
        ),
        interpretation=self.semantic_summary()
    )
```

## Benefits

1. **Type Safety**: No more `KeyError` or missing fields
2. **Validation**: Pydantic validates all data automatically
3. **Nested Structure**: Complex data relationships are clear
4. **Extensibility**: Easy to add new fields with backward compatibility

## Trade-offs

This approach adds some initial complexity:

- More model definitions upfront
- Need to maintain type consistency

However, the benefits far outweigh the costs, especially as the system grows.

## Conclusion

Typed contexts maintain CPA's philosophy of "models that understand themselves" while adding the safety and tooling benefits of full type coverage. This is the best of both worlds: simplicity where it matters, safety where it counts.
