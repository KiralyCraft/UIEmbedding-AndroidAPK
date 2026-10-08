FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home --uid 10001 collector
COPY server/collector /app/collector
USER collector
EXPOSE 8000
# No arbitrary X-Forwarded-* trust; override FORWARDED_ALLOW_IPS only for your proxy.
CMD ["uvicorn", "collector.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers", "--no-access-log"]
