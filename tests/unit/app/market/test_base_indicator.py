"""
Tests for BaseIndicator and IndicatorRegistry.

Tests the base class system and dynamic indicator registration.
"""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from src.market.analysis.base import AgentContext, BaseIndicator
from src.market.analysis.contexts import (
    BaseAgentContext,
    SignalSuggestion,
)
from src.market.analysis.registry import IndicatorRegistry


# Test implementation of BaseIndicator
class MockIndicator(BaseIndicator):
    """Mock indicator for testing base class."""

    test_value: float

    def semantic_summary(self) -> str:
        """One-line summary."""
        return f"Test indicator: {self.test_value}"

    def to_agent_context(self) -> AgentContext:
        """Agent context."""

        return BaseAgentContext(
            indicator="test", interpretation=f"Test indicator value: {self.test_value}"
        )

    def suggest_signal(self) -> SignalSuggestion:
        """Signal suggestion."""
        return SignalSuggestion(
            bias="neutral", strength="weak", reason="test", action="wait"
        )


class TestBaseIndicator:
    """Test BaseIndicator abstract class functionality."""

    # No need for registry state management in BaseIndicator tests
    # as we don't test registration here

    def test_base_indicator_is_abstract(self) -> None:
        """Test that BaseIndicator cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseIndicator(  # type: ignore
                symbol="TEST",
                timestamp=datetime.now(UTC),
            )

    def test_concrete_indicator_creation(self) -> None:
        """Test creating a concrete indicator."""
        now = datetime.now(UTC)
        indicator = MockIndicator(
            symbol="BTC-USD",
            timestamp=now,
            test_value=42.0,
        )

        assert indicator.symbol == "BTC-USD"
        assert indicator.timestamp == now
        assert indicator.test_value == 42.0

    def test_indicator_is_frozen(self) -> None:
        """Test that indicators are immutable."""
        indicator = MockIndicator(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            test_value=42.0,
        )

        with pytest.raises(ValidationError):
            indicator.test_value = 43.0  # type: ignore

    def test_required_methods(self) -> None:
        """Test that required methods are implemented."""
        indicator = MockIndicator(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            test_value=42.0,
        )

        # Test semantic_summary
        summary = indicator.semantic_summary()
        assert isinstance(summary, str)
        assert "Test indicator: 42.0" in summary

        # Test to_agent_context
        context = indicator.to_agent_context()

        assert isinstance(context, BaseAgentContext)
        assert context.indicator == "test"
        assert "42.0" in context.interpretation

        # Test suggest_signal
        signal = indicator.suggest_signal()
        assert isinstance(signal, SignalSuggestion)
        assert signal.bias == "neutral"
        assert signal.action == "wait"


class TestIndicatorRegistry:
    """Test IndicatorRegistry functionality."""

    @pytest.fixture
    def registry(self) -> IndicatorRegistry:
        """Create isolated test registry that allows all indicators."""
        from src.market.analysis.registry import RegistryConfig

        # Create config that enables all test indicators
        config = RegistryConfig(
            enabled_indicators=[
                "test_indicator",
                "test_get",
                "duplicate",
                "invalid",
                "creatable",
                "zebra",
                "alpha",
                "to_clear",
                "error_indicator",
            ]
        )
        return IndicatorRegistry(config=config)

    def test_register_indicator(self, registry: IndicatorRegistry) -> None:
        """Test registering an indicator."""

        @registry.register("test_indicator")
        class TestRegisteredIndicator(BaseIndicator):
            value: float

            def semantic_summary(self) -> str:
                return f"Value: {self.value}"

            def to_agent_context(self) -> dict[str, Any]:
                return {"value": self.value}

            def suggest_signal(self) -> dict[str, str]:
                return {
                    "bias": "neutral",
                    "strength": "weak",
                    "reason": "test",
                    "action": "wait",
                }

        # Check it's registered
        assert registry.has("test_indicator")
        assert "test_indicator" in registry.list_indicators()

    def test_get_indicator_class(self, registry: IndicatorRegistry) -> None:
        """Test retrieving an indicator class."""

        @registry.register("test_get")
        class TestGetIndicator(BaseIndicator):
            value: float

            def semantic_summary(self) -> str:
                return ""

            def to_agent_context(self) -> dict[str, Any]:
                return {}

            def suggest_signal(self) -> dict[str, str]:
                return {
                    "bias": "neutral",
                    "strength": "weak",
                    "reason": "test",
                    "action": "wait",
                }

        # Get the class
        indicator_class = registry.get("test_get")
        assert indicator_class is TestGetIndicator

    def test_get_nonexistent_indicator(self, registry: IndicatorRegistry) -> None:
        """Test getting a non-existent indicator."""
        with pytest.raises(KeyError, match="Indicator 'nonexistent' not found"):
            registry.get("nonexistent")

    def test_duplicate_registration(self, registry: IndicatorRegistry) -> None:
        """Test that duplicate registration raises error."""

        @registry.register("duplicate")
        class FirstIndicator(BaseIndicator):
            value: float

            def semantic_summary(self) -> str:
                return ""

            def to_agent_context(self) -> dict[str, Any]:
                return {}

            def suggest_signal(self) -> dict[str, str]:
                return {
                    "bias": "neutral",
                    "strength": "weak",
                    "reason": "test",
                    "action": "wait",
                }

        with pytest.raises(
            ValueError, match="Indicator 'duplicate' already registered"
        ):

            @registry.register("duplicate")
            class SecondIndicator(BaseIndicator):
                value: float

                def semantic_summary(self) -> str:
                    return ""

                def to_agent_context(self) -> dict[str, Any]:
                    return {}

                def suggest_signal(self) -> dict[str, str]:
                    return {
                        "bias": "neutral",
                        "strength": "weak",
                        "reason": "test",
                        "action": "wait",
                    }

    def test_invalid_registration(self, registry: IndicatorRegistry) -> None:
        """Test that non-BaseIndicator classes cannot be registered."""

        # Since we removed the BaseIndicator check, this test is no longer applicable
        # The registry now accepts any class - validation is done at usage time
        @registry.register("invalid")
        class NotAnIndicator:  # type: ignore
            pass

        # Should register without error
        assert registry.has("invalid")

    def test_create_indicator(self, registry: IndicatorRegistry) -> None:
        """Test creating an indicator through the registry."""

        @registry.register("creatable")
        class CreatableIndicator(BaseIndicator):
            value: float = 100.0

            def semantic_summary(self) -> str:
                return f"Value: {self.value}"

            def to_agent_context(self) -> dict[str, Any]:
                return {"value": self.value}

            def suggest_signal(self) -> dict[str, str]:
                return {
                    "bias": "neutral",
                    "strength": "weak",
                    "reason": "test",
                    "action": "wait",
                }

        # Create through registry
        indicator = registry.create(
            "creatable",
            symbol="TEST",
            timestamp=datetime.now(UTC),
            value=42.0,
        )

        assert isinstance(indicator, CreatableIndicator)
        assert indicator.value == 42.0

    def test_list_indicators_sorted(self, registry: IndicatorRegistry) -> None:
        """Test that list_indicators returns sorted names."""

        # Register in non-alphabetical order
        @registry.register("zebra")
        class ZebraIndicator(BaseIndicator):
            value: float

            def semantic_summary(self) -> str:
                return ""

            def to_agent_context(self) -> dict[str, Any]:
                return {}

            def suggest_signal(self) -> dict[str, str]:
                return {
                    "bias": "neutral",
                    "strength": "weak",
                    "reason": "test",
                    "action": "wait",
                }

        @registry.register("alpha")
        class AlphaIndicator(BaseIndicator):
            value: float

            def semantic_summary(self) -> str:
                return ""

            def to_agent_context(self) -> dict[str, Any]:
                return {}

            def suggest_signal(self) -> dict[str, str]:
                return {
                    "bias": "neutral",
                    "strength": "weak",
                    "reason": "test",
                    "action": "wait",
                }

        indicators = registry.list_indicators()
        assert indicators == ["alpha", "zebra"]

    def test_clear_registry(self, registry: IndicatorRegistry) -> None:
        """Test clearing the registry."""

        @registry.register("to_clear")
        class ToClearIndicator(BaseIndicator):
            value: float

            def semantic_summary(self) -> str:
                return ""

            def to_agent_context(self) -> dict[str, Any]:
                return {}

            def suggest_signal(self) -> dict[str, str]:
                return {
                    "bias": "neutral",
                    "strength": "weak",
                    "reason": "test",
                    "action": "wait",
                }

        assert registry.has("to_clear")

        registry.clear()

        assert not registry.has("to_clear")
        assert len(registry.list_indicators()) == 0
