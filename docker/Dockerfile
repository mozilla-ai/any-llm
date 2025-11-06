FROM python:3.13-slim as base

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

ARG VERSION=0.0.0+docker
ENV SETUPTOOLS_SCM_PRETEND_VERSION="${VERSION}"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .[all,gateway]

RUN useradd -m -u 1000 gateway && \
    chown -R gateway:gateway /app

USER gateway

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENV GATEWAY_HOST=0.0.0.0
ENV GATEWAY_PORT=8000

CMD ["any-llm-gateway", "serve"]
