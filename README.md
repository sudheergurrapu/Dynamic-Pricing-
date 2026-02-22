# Dynamic Pricing System for Online Stores

A full-stack IEEE demo-ready application that combines supervised learning and reinforcement learning to optimize product prices in real time.

## Stack
- **Backend:** FastAPI, scikit-learn, XGBoost (optional fallback), pandas
- **Frontend:** HTML/CSS/JavaScript + Chart.js
- **Deployment:** Docker + docker-compose

## Features
- `/predict` REST endpoint for optimized price, revenue prediction, confidence score, and KPI projections
- Data preprocessing pipeline with imputation, scaling, and one-hot encoding
- Feature engineering for demand trend, elasticity, competitor delta, and seasonality
- Model comparison: Random Forest, Gradient Boosting, XGBoost
- Hyperparameter tuning + cross-validation + best model auto-selection
- Q-learning agent for bounded pricing adjustment under business constraints
- Auto-refresh dashboard simulation and static vs dynamic price chart
- Swagger docs at `/docs`

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## Docker
```bash
docker compose up --build
```

## API example
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "base_price": 100,
    "demand_index": 0.8,
    "competitor_price": 95,
    "inventory_level": 120,
    "customer_segment": "regular",
    "month": 11,
    "day_of_week": 5,
    "historical_demand": [0.6, 0.65, 0.7, 0.75, 0.8]
  }'
```
