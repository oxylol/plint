# CNC Twin: a physics twin of machine, controller, tool, workpiece, chips and coolant (see README.md).
FROM python:3.12-slim

# WeasyPrint (PDF) needs Pango/HarfBuzz; OpenCASCADE (STEP import) needs the X/GL client libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libharfbuzz-subset0 libfontconfig1 fonts-dejavu-core \
        libgl1 libglu1-mesa libxrender1 libxext6 libsm6 libx11-6 libxi6 libxcursor1 libxrandr2 libxinerama1 \
        libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY cnctwin ./cnctwin
RUN pip install --no-cache-dir . && pip install --no-cache-dir numba-cuda || true
COPY db ./db
COPY web ./web
COPY examples ./examples
COPY validation ./validation
COPY docs ./docs

# Deterministic numerics (docs/determinism.md) and a pre-built kernel cache.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    NUMBA_CACHE_DIR=/app/.numba-cache CNCTWIN_ROOT=/app DATA_DIR=/data PORT=5210
RUN python -m cnctwin.cli warmup && chmod -R a+rX /app/.numba-cache

USER 1002:1002
EXPOSE 5210
HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5210/api/health', timeout=3)"
CMD ["python", "-m", "cnctwin.cli", "serve", "--host", "0.0.0.0", "--port", "5210"]
