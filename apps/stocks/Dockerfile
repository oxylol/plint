# Stocks on Plint: stock-trading models on paper, side by side (see README.md).
FROM python:3.12-slim
# lightgbm needs OpenMP at run time
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
# the CPU build of torch: Plint has no usable GPU, and the default wheel drags in 2 GB of CUDA
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY stocks ./stocks
COPY static ./static
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DATA_DIR=/data CACHE_DIR=/cache PORT=8000
EXPOSE 8000
CMD ["python", "-m", "stocks"]
