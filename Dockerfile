FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY SUMMARY_PROMPT.md .
RUN useradd --create-home --uid 10001 dochat
USER dochat
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "10000"]
