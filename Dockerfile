# ControlPlane.ai — container image (serves the Streamlit web UI).
# The UI runs the full decision pipeline in-process; no separate API
# service is required for the demo. To serve the FastAPI backend instead,
# override CMD with:  uvicorn api.app:app --host 0.0.0.0 --port $PORT
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CONTROLPLANE_UI_PORT=8501

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-generate the synthetic datasets at build time (deterministic).
RUN python run_generator.py

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8501/_stcore/health'); " || exit 1

# Respect a platform-provided $PORT (Render/Railway/HF Spaces) if set.
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
