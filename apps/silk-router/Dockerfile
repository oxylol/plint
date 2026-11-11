# silk-router on Plint: one Silk Compute key behind OpenAI- and Anthropic-shaped endpoints.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY router.py ./
USER nobody
EXPOSE 4141
CMD ["uvicorn", "router:app", "--host", "0.0.0.0", "--port", "4141", "--no-access-log"]
