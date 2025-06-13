"""
Modern indicator registry with dependency injection and Pydantic configuration.

This module provides a clean, testable registry pattern that aligns with
our protocol-first architecture and Pydantic philosophy.
"""

from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

T = TypeVar("T")


class RegistryConfig(BaseSettings):
    """Registry configuration using Pydantic Settings."""

    enabled_indicators: list[str] = Field(
        default=["rsi", "macd", "stochastic", "volume"],
        description="List of enabled indicator names",
    )

    model_config = {"env_prefix": "INDICATOR_REGISTRY_"}


class IndicatorRegistry(BaseModel):
    """
    Modern registry that's injectable and configurable.

    Follows Pydantic philosophy - the model IS the configuration.
    Provides clean dependency injection and test isolation.
    """

    config: RegistryConfig = Field(default_factory=RegistryConfig)
    indicators: dict[str, type] = Field(default_factory=dict, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def default(cls) -> "IndicatorRegistry":
        """
        Get default registry.

        Used for production scenarios where indicators register themselves.
        """
        # Use the singleton getter which handles creation
        global _default_registry
        if _default_registry is None:
            _default_registry = cls()
        return _default_registry

    @classmethod
    def testing(cls) -> "IndicatorRegistry":
        """
        Get isolated registry for testing.

        Auto-discovery is disabled to allow explicit control.
        """
        return cls()

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """
        Register an indicator with this registry instance.

        Only registers if the indicator is in the enabled list.

        Args:
            name: Unique name for the indicator

        Returns:
            Decorator function

        """

        def decorator(indicator_class: type[T]) -> type[T]:
            if name in self.config.enabled_indicators:
                if name in self.indicators:
                    raise ValueError(f"Indicator '{name}' already registered")

                self.indicators[name] = indicator_class

            return indicator_class

        return decorator

    def get(self, name: str) -> type:
        """
        Get an indicator class by name.

        Args:
            name: Registered indicator name

        Returns:
            Indicator class

        Raises:
            KeyError: If indicator not found

        """
        if name not in self.indicators:
            raise KeyError(f"Indicator '{name}' not found in registry")
        return self.indicators[name]

    def has(self, name: str) -> bool:
        """
        Check if an indicator is registered.

        Args:
            name: Indicator name to check

        Returns:
            True if registered, False otherwise

        """
        return name in self.indicators

    def list_indicators(self) -> list[str]:
        """
        List all registered indicator names.

        Returns:
            Sorted list of indicator names

        """
        return sorted(self.indicators.keys())

    def clear(self) -> None:
        """Clear all registered indicators (mainly for testing)."""
        self.indicators.clear()

    def create(self, name: str, **kwargs: object) -> object:
        """
        Create an indicator instance by name.

        Args:
            name: Registered indicator name
            **kwargs: Arguments to pass to indicator constructor

        Returns:
            Indicator instance

        Raises:
            KeyError: If indicator not found

        """
        indicator_class = self.get(name)
        return indicator_class(**kwargs)


# Default instance for backward compatibility
_default_registry: IndicatorRegistry | None = None


def get_default_registry() -> IndicatorRegistry:
    """Get or create the default registry singleton."""
    global _default_registry
    if _default_registry is None:
        _default_registry = IndicatorRegistry.default()
    return _default_registry


def register(name: str) -> Callable[[type[T]], type[T]]:
    """
    Register an indicator with the default registry.

    This maintains the simple decorator API for common use cases.

    Args:
        name: Unique name for the indicator

    Returns:
        Decorator function

    Example:
        @register("rsi")
        class RSIAnalysis(BaseIndicator):
            ...

    """
    return get_default_registry().register(name)
