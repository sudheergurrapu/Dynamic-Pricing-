"""Data generation, preprocessing, and feature engineering utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLS = [
    "base_price",
    "demand_index",
    "competitor_price",
    "inventory_level",
    "demand_trend",
    "price_elasticity",
    "competitor_diff",
    "seasonal_factor",
]
CATEGORICAL_COLS = ["customer_segment", "month", "day_of_week"]


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing pipeline for missing values, scaling, and encoding."""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )


def seasonal_factor(month: int) -> float:
    """Approximate seasonal effect for retail cycles."""
    return 1.0 + 0.2 * np.sin((month - 1) / 12.0 * 2 * np.pi)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features required by the pricing model."""
    df = frame.copy()
    df["demand_trend"] = (df["demand_index"] - 0.75).fillna(0.0)
    df["price_elasticity"] = -1.1 + 0.5 * (df["competitor_price"] - df["base_price"]) / df["base_price"].clip(lower=1)
    df["competitor_diff"] = df["base_price"] - df["competitor_price"]
    df["seasonal_factor"] = df["month"].apply(seasonal_factor)
    return df


def generate_training_data(size: int = 1800, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic-yet-realistic training samples for demonstration."""
    rng = np.random.default_rng(seed)

    base_price = rng.uniform(20, 400, size)
    demand_index = rng.uniform(0.4, 1.6, size)
    competitor_price = base_price * rng.uniform(0.75, 1.2, size)
    inventory_level = rng.integers(20, 1000, size)
    customer_segment = rng.choice(["new", "regular", "vip"], size=size, p=[0.3, 0.5, 0.2])
    month = rng.integers(1, 13, size)
    day_of_week = rng.integers(0, 7, size)

    df = pd.DataFrame(
        {
            "base_price": base_price,
            "demand_index": demand_index,
            "competitor_price": competitor_price,
            "inventory_level": inventory_level,
            "customer_segment": customer_segment,
            "month": month,
            "day_of_week": day_of_week,
        }
    )
    df = engineer_features(df)

    segment_boost = pd.Series(customer_segment).map({"new": 0.92, "regular": 1.0, "vip": 1.12}).to_numpy()
    seasonal = df["seasonal_factor"].to_numpy()

    demand_units = 40 * demand_index * segment_boost * seasonal * np.exp(-base_price / competitor_price)
    demand_units *= (1 + np.clip((200 - inventory_level) / 500, -0.2, 0.3))
    noise = rng.normal(0, 3.0, size)
    demand_units = np.maximum(demand_units + noise, 1)
    revenue = demand_units * base_price

    missing_mask = rng.random(size) < 0.04
    df.loc[missing_mask, "competitor_price"] = np.nan
    df.loc[rng.random(size) < 0.03, "customer_segment"] = None

    return df, pd.Series(revenue, name="revenue")
