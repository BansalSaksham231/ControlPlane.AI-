# ControlPlane.ai — Streamlit Enterprise Command Center (production image).
#
#   docker build -f ui.Dockerfile -t controlplane-ui .
#   docker run -p 8501:8501 -e CONTROLPLANE_API_URL=http://api:8000 controlplane-ui
#
# The UI runs the decision pipeline in-process for the interactive demo and,
# where wired, calls the API service at CONTROLPLANE_API_URL.

# ---- build stage ---------------------------------------------------------
FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ---- runtime stage ------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONTROLPLANE_UI_PORT=8501

RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app
COPY --from=build /install /usr/local
COPY . .

RUN python run_generator.py && chown -R app:app /app

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
