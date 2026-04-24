from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from typing_extensions import override

from any_llm.gateway import __version__
from any_llm.gateway.api.deps import set_config
from any_llm.gateway.api.main import register_routers
from any_llm.gateway.core.config import GatewayConfig
from any_llm.gateway.core.database import get_db, init_db
from any_llm.gateway.rate_limit import RateLimiter
from any_llm.gateway.services.bootstrap_service import bootstrap_first_api_key
from any_llm.gateway.services.pricing_init_service import initialize_pricing_from_config

_PUBLIC_PREFIXES = ("/health",)

_ROOT_TUTORIAL_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AI Gateway</title>
    <style>
      body {
        background: #efefef;
        color: #111827;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
        margin: 0;
      }
      main {
        margin: 16px auto;
        max-width: 860px;
        padding: 0 16px;
        line-height: 1.5;
      }
      h1 {
        font-size: 32px;
        margin: 6px 0 10px;
      }
      .sub {
        font-size: 20px;
        margin-bottom: 16px;
      }
      .link {
        font-size: 20px;
        font-weight: 700;
        color: #0f62fe;
        text-decoration: underline;
      }
      .note {
        font-size: 16px;
      }
      .block {
        background: #e5e7eb;
        border-radius: 2px;
        margin: 18px 0;
        padding: 18px;
      }
      pre {
        margin: 0;
        overflow-x: auto;
      }
      code {
        color: #111827;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 14px;
      }
    </style>
  </head>
  <body>
    <main>
      <h1>AI Gateway (Proxy Server)</h1>
      <div class="sub">
        <a class="link" href="https://docs.mozilla.ai/any-llm/gateway/quickstart/">Gateway Quickstart</a>
      </div>

      <p class="note">On first startup, the gateway prints a bootstrap API key in logs. Use that value as <code>YOUR_GATEWAY_KEY</code>.</p>

      <div class="block">
        <pre><code>pip install openai
</code></pre>
      </div>

      <div class="block">
        <pre><code>from openai import OpenAI

client = OpenAI(
    api_key="YOUR_GATEWAY_KEY",
    base_url="http://0.0.0.0:8000/v1",
)

response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response.choices[0].message.content)
</code></pre>
      </div>
    </main>
  </body>
</html>
"""


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses.

    Sets standard security headers on every response, plus cache-control
    headers on non-health endpoints to prevent CDN/proxy caches from
    storing authenticated responses.
    """

    @override
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if not request.url.path.startswith(_PUBLIC_PREFIXES):
            response.headers["Cache-Control"] = "private, no-store, no-cache"
            response.headers["Vary"] = "Authorization"
        return response


def create_app(config: GatewayConfig) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        config: Gateway configuration

    Returns:
        Configured FastAPI application

    """
    init_db(config.database_url, auto_migrate=config.auto_migrate)
    set_config(config)

    db = next(get_db())
    try:
        bootstrap_first_api_key(config, db)
        initialize_pricing_from_config(config, db)
    finally:
        db.close()

    app = FastAPI(
        title="any-llm-gateway",
        description="A clean FastAPI gateway for any-llm with API key management",
        version=__version__,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root_tutorial() -> str:
        return _ROOT_TUTORIAL_HTML

    app.add_middleware(SecurityHeadersMiddleware)

    if config.cors_allow_origins:
        allow_credentials = "*" not in config.cors_allow_origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_allow_origins,
            allow_credentials=allow_credentials,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization", "X-AnyLLM-Key", "x-api-key"],
        )

    if config.enable_metrics:
        from any_llm.gateway.metrics import MetricsMiddleware

        app.add_middleware(MetricsMiddleware)

    if config.rate_limit_rpm is not None:
        app.state.rate_limiter = RateLimiter(config.rate_limit_rpm)
    else:
        app.state.rate_limiter = None

    register_routers(app)

    if config.enable_metrics:
        from any_llm.gateway.metrics import metrics_endpoint

        app.add_route("/metrics", metrics_endpoint, methods=["GET"])

    return app
