# ControlPlane.ai — FastAPI backend (production image).
#
# Multi-stage: a build stage compiles/installs dependencies into an isolated
# prefix, the runtime stage copies only that prefix + the app and runs as a
# non-root user under Uvicorn.
#
#   docker build -f api.Dockerfile -t controlplane-api .
#   docker run -p 8000:8000 -e CONTROLPLANE_PERSISTENCE=1 controlplane-api

# ---- build stage ---------------------------------------------------------
FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install into a relocatable prefix so the runtime stage can copy it wholesale.
COPY requirements.txt requirements-db.txt ./
RUN pip install --prefix=/install -r requirements.txt \
 && pip install --prefix=/install -r requirements-db.txt

# ---- runtime stage ------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONTROLPLANE_PERSISTENCE=1 \
    CONTROLPLANE_FAST_START=0

# psycopg[binary] ships its own libpq, so no system libpq is required.
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app

COPY --from=build /install /usr/local
COPY . .

# Deterministic synthetic datasets baked at build time.
RUN python run_generator.py && chown -R app:app /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# One worker keeps the in-process audit cache coherent; scale horizontally
# with the shared database instead of adding workers.
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS:-1}"]
