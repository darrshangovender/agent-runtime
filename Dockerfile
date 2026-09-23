# --- build stage: install the package into an isolated prefix -----------------
FROM python:3.11-slim AS build

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY agent_runtime ./agent_runtime
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install .

# --- runtime stage: slim image, non-root, examples only -----------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AGENT_RUNTIME_DB=/data/checkpoints.db

RUN groupadd --system agent && useradd --system --gid agent --create-home agent \
    && mkdir -p /data && chown agent:agent /data

COPY --from=build /install /usr/local
WORKDIR /app
COPY --chown=agent:agent examples ./examples

USER agent
VOLUME ["/data"]

CMD ["python", "examples/research_agent.py"]
