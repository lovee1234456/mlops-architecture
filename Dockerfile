FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements-api.txt
COPY src/ src/
COPY models/ models/
COPY evaluation/ evaluation/
COPY monitoring/ monitoring/

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]