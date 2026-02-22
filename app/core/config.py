"""Application configuration values."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    min_price_factor: float = 0.70
    max_price_factor: float = 1.50
    max_step_volatility: float = 0.08
    refresh_interval_ms: int = 5000


settings = Settings()
