# Minimal container for Streamlit on Cloud Run / Render
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Cloud Run sets $PORT; default to 8080 locally
ENV PORT=8080
EXPOSE 8080

CMD ["streamlit", "run", "evms-agents-app.py", "--server.port", "${PORT}", "--server.address", "0.0.0.0"]

