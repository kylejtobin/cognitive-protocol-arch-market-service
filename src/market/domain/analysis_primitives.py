"""
Domain primitives for technical analysis indicators.

These primitives provide type safety and rich behavior for analysis values,
ensuring indicator values are always valid and providing domain-specific operations.
"""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .primitives import Percentage


class RSIValue(BaseModel):
    """
    Relative Strength Index value with domain logic.

    RSI is always between 0 and 100, with specific zones:
    - 0-30: Oversold
    - 30-70: Neutral
    - 70-100: Overbought
    """

    model_config = ConfigDict(frozen=True)

    value: Decimal = Field(ge=0, le=100, description="RSI value between 0 and 100")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def zone(cls) -> Literal["oversold", "neutral", "overbought"]:
        """Determine RSI zone based on standard thresholds."""
        if cls.value < 30:
            return "oversold"
        elif cls.value > 70:
            return "overbought"
        return "neutral"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def strength(cls) -> Literal["extreme", "strong", "moderate", "weak"]:
        """Determine strength of the RSI signal."""
        if cls.value < 20 or cls.value > 80:
            return "extreme"
        elif cls.value < 30 or cls.value > 70:
            return "strong"
        elif cls.value < 40 or cls.value > 60:
            return "moderate"
        return "weak"

    def distance_from_neutral(self) -> Decimal:
        """Calculate distance from neutral (50) level."""
        return abs(self.value - Decimal("50"))

    def distance_from_zone_boundary(self) -> Decimal:
        """Calculate distance from nearest overbought/oversold boundary."""
        if self.value < 30:
            return Decimal("30") - self.value  # Distance to exit oversold
        elif self.value > 70:
            return self.value - Decimal("70")  # Distance into overbought
        else:
            # In neutral zone, return distance to nearest boundary
            to_oversold = abs(self.value - Decimal("30"))
            to_overbought = abs(self.value - Decimal("70"))
            return min(to_oversold, to_overbought)

    def to_decimal(self) -> Decimal:
        """Convert to Decimal for protocol compatibility."""
        return self.value

    def format_display(self) -> str:
        """Format for display with zone indication."""
        zone_indicator = {"oversold": "↓", "neutral": "→", "overbought": "↑"}[self.zone]

        return f"RSI {self.value:.1f} {zone_indicator}"

    def __str__(self) -> str:
        """String representation."""
        return f"RSI({self.value:.1f}, {self.zone})"


class MACDValue(BaseModel):
    """
    MACD (Moving Average Convergence Divergence) value with components.

    Consists of:
    - MACD line: 12-period EMA - 26-period EMA
    - Signal line: 9-period EMA of MACD line
    - Histogram: MACD line - Signal line
    """

    model_config = ConfigDict(frozen=True)

    macd_line: Decimal = Field(description="MACD line value")
    signal_line: Decimal = Field(description="Signal line value")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def histogram(cls) -> Decimal:
        """MACD histogram (MACD - Signal)."""
        return cls.macd_line - cls.signal_line

    @computed_field  # type: ignore[prop-decorator]
    @property
    def trend(cls) -> Literal["bullish", "bearish", "neutral"]:
        """Determine trend based on histogram."""
        if cls.histogram > Decimal("0.5"):
            return "bullish"
        elif cls.histogram < Decimal("-0.5"):
            return "bearish"
        return "neutral"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_positive(cls) -> bool:
        """Whether MACD line is above zero."""
        return cls.macd_line > 0

    def crossover_strength(self) -> Percentage:
        """
        Calculate crossover strength as percentage.

        Stronger divergence between lines indicates stronger signal.
        """
        if self.signal_line == 0:
            return Percentage(value=Decimal("0"))

        divergence = abs(self.histogram / self.signal_line) * 100
        # Cap at 100% for extreme divergences
        return Percentage(value=min(divergence, Decimal("100")))

    def momentum_direction(self) -> Literal["accelerating", "decelerating", "stable"]:
        """Determine if momentum is accelerating or decelerating."""
        hist_abs = abs(self.histogram)
        if hist_abs < Decimal("0.1"):
            return "stable"
        # Positive histogram and MACD above signal = bullish acceleration
        elif self.histogram > 0:
            return "accelerating"
        # Negative histogram and MACD below signal = bearish acceleration
        else:
            return "decelerating"

    def format_display(self) -> str:
        """Format for display."""
        trend_indicator = {"bullish": "↑", "bearish": "↓", "neutral": "→"}[self.trend]

        return f"MACD {self.macd_line:.2f}/{self.signal_line:.2f} {trend_indicator}"

    def __str__(self) -> str:
        """String representation."""
        return f"MACD(line={self.macd_line:.2f}, signal={self.signal_line:.2f}, hist={self.histogram:.2f})"


class StochasticValue(BaseModel):
    """
    Stochastic oscillator values (%K and %D).

    Both values range from 0 to 100:
    - %K: Fast stochastic
    - %D: Slow stochastic (smoothed %K)

    Zones:
    - 0-20: Oversold
    - 20-80: Neutral
    - 80-100: Overbought
    """

    model_config = ConfigDict(frozen=True)

    k_value: Decimal = Field(ge=0, le=100, description="%K value")
    d_value: Decimal = Field(ge=0, le=100, description="%D value")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def zone(cls) -> Literal["oversold", "neutral", "overbought"]:
        """Determine zone based on %K value."""
        if cls.k_value < 20:
            return "oversold"
        elif cls.k_value >= 80:  # Changed to >= for 80 to be overbought
            return "overbought"
        return "neutral"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def momentum(cls) -> Literal["increasing", "decreasing", "stable"]:
        """Determine momentum based on K vs D relationship."""
        diff = cls.k_value - cls.d_value
        if abs(diff) < Decimal("2"):
            return "stable"
        elif diff > 0:
            return "increasing"
        return "decreasing"

    def divergence(self) -> Decimal:
        """Calculate divergence between %K and %D."""
        return abs(self.k_value - self.d_value)

    def is_crossing(self, threshold: Decimal = Decimal("1")) -> bool:
        """Check if %K and %D are crossing (very close)."""
        return self.divergence() <= threshold

    def format_display(self) -> str:
        """Format for display."""
        momentum_indicator = {"increasing": "↑", "decreasing": "↓", "stable": "→"}[
            self.momentum
        ]

        return f"Stoch %K:{self.k_value:.1f}/%D:{self.d_value:.1f} {momentum_indicator}"

    def __str__(self) -> str:
        """String representation."""
        return f"Stochastic(K={self.k_value:.1f}, D={self.d_value:.1f}, {self.zone})"


class VolumeProfile(BaseModel):
    """
    Volume profile with buy/sell pressure analysis.

    Provides rich volume analysis beyond simple volume values.
    """

    model_config = ConfigDict(frozen=True)

    current_volume: Decimal = Field(ge=0, description="Current period volume")
    average_volume: Decimal = Field(ge=0, description="Average volume over period")
    buy_volume: Decimal = Field(ge=0, description="Volume on upticks")
    sell_volume: Decimal = Field(ge=0, description="Volume on downticks")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def volume_ratio(cls) -> Decimal:
        """Ratio of current to average volume."""
        if cls.average_volume == 0:
            return Decimal("0")
        return cls.current_volume / cls.average_volume

    @computed_field  # type: ignore[prop-decorator]
    @property
    def buy_pressure(cls) -> Percentage:
        """Buy volume as percentage of total."""
        total = cls.buy_volume + cls.sell_volume
        if total == 0:
            return Percentage(value=Decimal("50"))  # Neutral if no volume
        return Percentage(value=(cls.buy_volume / total) * 100)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sell_pressure(cls) -> Percentage:
        """Sell volume as percentage of total."""
        total = cls.buy_volume + cls.sell_volume
        if total == 0:
            return Percentage(value=Decimal("50"))  # Neutral if no volume
        return Percentage(value=(cls.sell_volume / total) * 100)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pressure_balance(cls) -> Literal["buy_heavy", "sell_heavy", "balanced"]:
        """Determine overall pressure balance."""
        buy_pct = cls.buy_pressure.value
        if buy_pct > 60:
            return "buy_heavy"
        elif buy_pct < 40:
            return "sell_heavy"
        return "balanced"

    def is_abnormal(self, threshold: Decimal = Decimal("2")) -> bool:
        """Check if volume is abnormally high."""
        return self.volume_ratio > threshold

    def is_climactic(self) -> bool:
        """Check if volume shows climactic activity (very high with extreme pressure)."""
        is_high_volume = self.volume_ratio > Decimal("3")
        is_extreme_pressure = (
            self.buy_pressure.value > 80 or self.sell_pressure.value > 80
        )
        return is_high_volume and is_extreme_pressure

    def format_display(self) -> str:
        """Format for display."""
        if self.is_abnormal():
            volume_desc = f"{self.volume_ratio:.1f}x avg (HIGH)"
        else:
            volume_desc = f"{self.volume_ratio:.1f}x avg"

        pressure_indicator = {"buy_heavy": "↑", "sell_heavy": "↓", "balanced": "→"}[
            self.pressure_balance
        ]

        return f"Vol {volume_desc} {pressure_indicator}"

    def __str__(self) -> str:
        """String representation."""
        return f"Volume({self.volume_ratio:.1f}x, buy={self.buy_pressure.value:.0f}%)"


class BollingerBandPosition(BaseModel):
    """
    Position within Bollinger Bands.

    Tracks price position relative to upper, middle, and lower bands.
    """

    model_config = ConfigDict(frozen=True)

    price: Decimal = Field(gt=0, description="Current price")
    upper_band: Decimal = Field(gt=0, description="Upper band value")
    middle_band: Decimal = Field(gt=0, description="Middle band (SMA)")
    lower_band: Decimal = Field(gt=0, description="Lower band value")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def position_pct(cls) -> Percentage:
        """Position as percentage between bands (0% = lower, 100% = upper)."""
        band_width = cls.upper_band - cls.lower_band
        if band_width == 0:
            return Percentage(value=Decimal("50"))

        position = (cls.price - cls.lower_band) / band_width * 100
        # Can be outside 0-100 if price is outside bands
        return Percentage(value=position)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def band_width(cls) -> Decimal:
        """Width of the bands."""
        return cls.upper_band - cls.lower_band

    @computed_field  # type: ignore[prop-decorator]
    @property
    def band_width_pct(cls) -> Percentage:
        """Band width as percentage of middle band."""
        if cls.middle_band == 0:
            return Percentage(value=Decimal("0"))
        return Percentage(value=(cls.band_width / cls.middle_band) * 100)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def squeeze_detected(cls) -> bool:
        """Detect Bollinger Band squeeze (narrow bands)."""
        # Squeeze when bands are less than or equal to 2% of price
        return cls.band_width_pct.value <= 2

    @computed_field  # type: ignore[prop-decorator]
    @property
    def position_zone(
        cls,
    ) -> Literal["above_upper", "upper_half", "lower_half", "below_lower"]:
        """Determine position zone."""
        if cls.price > cls.upper_band:
            return "above_upper"
        elif cls.price >= cls.middle_band:  # Changed to >= for middle band
            return "upper_half"
        elif cls.price > cls.lower_band:
            return "lower_half"
        return "below_lower"

    def distance_from_band(self, band: Literal["upper", "middle", "lower"]) -> Decimal:
        """Calculate distance from specified band."""
        band_values = {
            "upper": self.upper_band,
            "middle": self.middle_band,
            "lower": self.lower_band,
        }
        return abs(self.price - band_values[band])

    def format_display(self) -> str:
        """Format for display."""
        if self.squeeze_detected:
            squeeze_text = " [SQUEEZE]"
        else:
            squeeze_text = ""

        zone_indicator = {
            "above_upper": "↑↑",
            "upper_half": "↑",
            "lower_half": "↓",
            "below_lower": "↓↓",
        }[self.position_zone]

        return f"BB {self.position_pct.value:.0f}% {zone_indicator}{squeeze_text}"

    def __str__(self) -> str:
        """String representation."""
        return f"BB({self.position_pct.value:.0f}%, {self.position_zone})"
