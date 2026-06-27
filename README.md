# Energy Demand Forecasting System

![CI](https://github.com/giulia-lelli/energy-forecast/actions/workflows/ci.yml/badge.svg)

End-to-end ML system for hourly energy demand forecasting on the PJM grid, with a production-ready REST API and Docker deployment.

## Results

| Metric | Naive Baseline | LightGBM | Improvement |
|--------|---------------|----------|-------------|
| MAE    | 1,625 MW      | 692 MW   | 57% better  |
| RMSE   | 2,372 MW      | 961 MW   | 60% better  |
| MAPE   | 5.84%         | 2.51%    | 57% better  |

## Project Structure
## Approach

**Data:** 3.75 years of hourly energy consumption (1998–2002) from the PJM regional grid (~33k observations). Handled 8 missing timestamps caused by DST spring-forward transitions via linear interpolation.

**Features (17 total):**
- Lag features: 24h, 48h, 168h (same hour yesterday/last week)
- Rolling statistics: mean and std over 24h and 168h windows
- Calendar features: hour, day of week, month, quarter, is_weekend
- Fourier terms: sine/cosine encoding of daily and weekly cycles

**Model:** LightGBM with early stopping. Lag features dominate importance (lag_24 accounts for ~2x the gain of any other feature), confirming the strong daily periodicity in energy consumption.

**Validation:** Strict temporal split — last 3 months held out as test set. No data leakage.

## Quick Start

### Run the API locally
```bash
pip install -r requirements.txt
uvicorn src.api:app --reload --port 8000
```

### Run with Docker
```bash
docker build -t energy-forecast .
docker run -p 8000:8000 energy-forecast
```

### Make a prediction
```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{"datetime": "2001-11-05 14:00:00"}'
```

### Train the model
```bash
cd src
python train.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root |
| GET | `/health` | Health check |
| POST | `/forecast` | Get energy forecast for a datetime |
| GET | `/docs` | Interactive API documentation |

## Stack
Python · Pandas · LightGBM · FastAPI · Docker · GitHub Actions
