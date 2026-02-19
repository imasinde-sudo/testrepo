# ── ETL Service ──────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# matplotlib needs these even in headless/Agg mode
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY test_etl_pipeline.py .

# ── Runtime directories ───────────────────────────────────────────────────────
# /app/data  → mount your input CSV here  (read-only is fine)
# /app/output → mount a host directory here to retrieve charts + CSV
RUN mkdir -p /app/data /app/output

# ── Environment variable defaults ─────────────────────────────────────────────
# Override these in docker-compose.yml or with -e flags at runtime
ENV INPUT_CSV=/app/data/input.csv \
    OUTPUT_DIR=/app/output \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["python", "test_etl_pipeline.py"]