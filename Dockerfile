# One image for both the API and the UI; docker-compose picks the command.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# pyproject.toml lists the packages (agentops*, api*, ui*, eval*), so the source
# must be present before `pip install .`
COPY . .
RUN pip install .

# Run as a normal user, not root. /data is where the optional SQLite file lives.
RUN useradd --create-home appuser && mkdir /data && chown appuser /data
USER appuser

EXPOSE 8000
# Hosts like Render inject $PORT; locally it falls back to 8000.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]