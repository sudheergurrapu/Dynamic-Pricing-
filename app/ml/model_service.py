"""Model training, evaluation, selection, and inference service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from app.ml.preprocessing import build_preprocessor, engineer_features, generate_training_data
from app.ml.rl_agent import QLearningPricer

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    metrics: dict[str, float]


class PricingModelService:
    """Owns full ML lifecycle used by API endpoints."""

    def __init__(self) -> None:
        self.preprocessor = build_preprocessor()
        self.rl_agent = QLearningPricer()
        self.trained_model = self._train_and_select()

    def _candidate_models(self) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
        candidates: dict[str, tuple[Any, dict[str, list[Any]]]] = {
            "RandomForest": (
                RandomForestRegressor(random_state=42),
                {"model__n_estimators": [120, 200], "model__max_depth": [6, 12]},
            ),
            "GradientBoosting": (
                GradientBoostingRegressor(random_state=42),
                {"model__n_estimators": [120, 200], "model__learning_rate": [0.05, 0.1]},
            ),
        }
        if XGBRegressor is not None:
            candidates["XGBoost"] = (
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    n_estimators=250,
                    max_depth=5,
                    learning_rate=0.06,
                    subsample=0.9,
                    colsample_bytree=0.9,
                ),
                {
                    "model__n_estimators": [180, 250],
                    "model__max_depth": [4, 5],
                    "model__learning_rate": [0.05, 0.08],
                },
            )
        return candidates

    def _train_and_select(self) -> TrainedModel:
        X, y = generate_training_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        best_model: TrainedModel | None = None
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        for name, (model, grid) in self._candidate_models().items():
            pipe = Pipeline([("preprocess", self.preprocessor), ("model", model)])
            search = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error", cv=3, n_jobs=-1)
            search.fit(X_train, y_train)

            final_pipe = search.best_estimator_
            pred = final_pipe.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            cv_r2 = float(np.mean(cross_val_score(final_pipe, X, y, cv=cv, scoring="r2", n_jobs=-1)))

            metrics = {"mae": float(mae), "mse": float(mse), "r2": float(r2), "cv_r2": cv_r2}
            trained = TrainedModel(name=name, pipeline=final_pipe, metrics=metrics)

            if best_model is None or trained.metrics["r2"] > best_model.metrics["r2"]:
                best_model = trained

        if best_model is None:
            raise RuntimeError("Failed to train any model")

        return best_model

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Produce pricing and KPI predictions with RL adjustment."""
        history = payload.pop("historical_demand", [payload["demand_index"]])
        demand_trend = float(history[-1] - history[0]) if len(history) > 1 else 0.0

        row = pd.DataFrame([payload])
        row = engineer_features(row)
        row["demand_trend"] = demand_trend

        revenue_pred = float(self.trained_model.pipeline.predict(row)[0])
        base_price = float(payload["base_price"])
        optimized_price = base_price * (1 + 0.35 * payload["demand_index"] - 0.00035 * payload["inventory_level"]) + 0.2 * (
            payload["competitor_price"] - base_price
        )

        bounded_price = self.rl_agent.adjust_price(
            model_price=float(optimized_price),
            base_price=base_price,
            demand_index=float(payload["demand_index"]),
            inventory_level=int(payload["inventory_level"]),
        )

        confidence = float(np.clip(self.trained_model.metrics["r2"], 0, 1))
        baseline_rev = base_price * payload["demand_index"] * 25
        uplift_pct = ((revenue_pred - baseline_rev) / max(baseline_rev, 1e-6)) * 100

        conversion_base = np.clip(1.0 - 0.3 * ((base_price - payload["competitor_price"]) / base_price), 0.2, 1.6)
        conversion_new = np.clip(1.0 - 0.3 * ((bounded_price - payload["competitor_price"]) / base_price), 0.2, 1.8)
        conversion_improvement = (conversion_new - conversion_base) / max(conversion_base, 1e-6) * 100

        kpis = {
            "revenue_growth": round(max(uplift_pct, -50), 2),
            "profit_margin": round(np.clip((bounded_price - 0.58 * base_price) / bounded_price * 100, 5, 75), 2),
            "conversion_rate": round(conversion_new * 100, 2),
            "inventory_turnover": round(np.clip(payload["demand_index"] * 2.5 / max(payload["inventory_level"] / 500, 0.2), 0.2, 8), 2),
        }

        return {
            "optimized_price": round(optimized_price, 2),
            "bounded_price": round(bounded_price, 2),
            "revenue_prediction": round(revenue_pred, 2),
            "confidence_score": round(confidence, 4),
            "expected_revenue_uplift_pct": round(uplift_pct, 2),
            "conversion_rate_improvement_pct": round(conversion_improvement, 2),
            "selected_model": self.trained_model.name,
            "model_metrics": self.trained_model.metrics,
            "kpis": kpis,
        }
