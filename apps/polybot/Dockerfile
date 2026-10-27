# Polybot on Plint: a Polymarket consensus follower (see README.md).
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY polybot ./polybot
COPY static ./static
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DATA_DIR=/data PORT=8000
EXPOSE 8000
CMD ["python", "-m", "polybot"]
