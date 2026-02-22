"""FastAPI entry point for the Dynamic Pricing System."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from app.api.routes import router as pricing_router
from app.core.config import settings

app = FastAPI(
    title="Dynamic Pricing System for Online Stores",
    description="ML + RL based pricing optimization API",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.include_router(pricing_router)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    """Serve the dashboard UI."""
    return templates.TemplateResponse("index.html", {"request": request, "refresh_ms": settings.refresh_interval_ms})
