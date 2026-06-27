FROM python:3.9-slim

WORKDIR /app

# Install system dependencies needed by LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model
COPY src/ ./src/
COPY models/ ./models/
COPY data/raw/PJM_Load_hourly.csv ./data/raw/PJM_Load_hourly.csv

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
