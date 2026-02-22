"""Pydantic schemas for pricing requests and responses."""

from typing import List, Literal

from pydantic import BaseModel, Field, conint, confloat


class PricingRequest(BaseModel):
    base_price: confloat(gt=0) = Field(..., description="Reference product price")
    demand_index: confloat(ge=0, le=2) = Field(..., description="Demand signal index")
    competitor_price: confloat(gt=0) = Field(..., description="Current competitor market price")
    inventory_level: conint(ge=0) = Field(..., description="Current stock level")
    customer_segment: Literal["new", "regular", "vip"] = "regular"
    month: conint(ge=1, le=12) = 1
    day_of_week: conint(ge=0, le=6) = 0
    historical_demand: List[confloat(ge=0, le=2)] = Field(default_factory=lambda: [0.8, 0.85, 0.9, 0.95, 1.0])


class KPIOutput(BaseModel):
    revenue_growth: float
    profit_margin: float
    conversion_rate: float
    inventory_turnover: float


class PricingResponse(BaseModel):
    optimized_price: float
    bounded_price: float
    revenue_prediction: float
    confidence_score: float
    expected_revenue_uplift_pct: float
    conversion_rate_improvement_pct: float
    selected_model: str
    model_metrics: dict
    kpis: KPIOutput
