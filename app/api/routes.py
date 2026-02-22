"""API routes for dynamic pricing predictions."""

from fastapi import APIRouter

from app.ml.model_service import PricingModelService
from app.schemas.pricing import PricingRequest, PricingResponse

router = APIRouter(tags=["pricing"])
service = PricingModelService()


@router.post("/predict", response_model=PricingResponse)
def predict_price(payload: PricingRequest) -> PricingResponse:
    """Return optimized pricing and KPI forecasts."""
    result = service.predict(payload.model_dump())
    return PricingResponse(**result)
