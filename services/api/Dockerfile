# syntax=docker/dockerfile:1.7
# Build context is the repository root.

ARG PYTHON_IMAGE=python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.32@sha256:df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c
ARG LWA_IMAGE=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.0@sha256:b4da35991627bdac98a81c377d0cc28e6989687359576dfda9f0b64be835d648

FROM ${UV_IMAGE} AS uv
FROM ${LWA_IMAGE} AS lwa

FROM ${PYTHON_IMAGE} AS models
WORKDIR /src
COPY artifacts.lock.json ./
COPY packages/speechgrade/src/speechgrade/__init__.py packages/speechgrade/src/speechgrade/artifacts.py speechgrade/
RUN python -m speechgrade.artifacts --lock artifacts.lock.json --out /models

FROM ${PYTHON_IMAGE} AS deps
ARG TARGETARCH
# praat-parselmouth publishes no linux/arm64 wheel, so arm64 compiles it from source.
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends build-essential cmake \
        && rm -rf /var/lib/apt/lists/*; \
    fi
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
COPY pyproject.toml uv.lock ./
COPY packages/speechgrade/pyproject.toml packages/speechgrade/
COPY services/api/pyproject.toml services/api/
COPY dataset_factory/pyproject.toml dataset_factory/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package oratio-api --no-install-workspace
COPY packages/speechgrade/src packages/speechgrade/src
COPY services/api/src services/api/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package oratio-api --no-editable

FROM ${PYTHON_IMAGE} AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home oratio
COPY --from=lwa /lambda-adapter /opt/extensions/lambda-adapter
COPY --from=deps /opt/venv /opt/venv
COPY --from=models /models /opt/oratio/models
COPY services/api/baselines /opt/oratio/baselines
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ORATIO_MODEL_DIR=/opt/oratio/models \
    AWS_LWA_PORT=8000 \
    AWS_LWA_READINESS_CHECK_PATH=/api/health \
    AWS_LWA_INVOKE_MODE=response_stream
USER oratio
EXPOSE 8000
CMD ["uvicorn", "oratio_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
