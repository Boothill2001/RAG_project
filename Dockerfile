# Advanced RAG API — production image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer — rebuilds only when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Models (~300MB) are downloaded on first start and cached in this volume,
# keeping the image small and rebuilds fast.
VOLUME ["/root/.cache/huggingface", "/app/index"]

EXPOSE 8000

# Build the index if missing, then serve.
CMD ["sh", "-c", "python scripts/build_index.py && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
