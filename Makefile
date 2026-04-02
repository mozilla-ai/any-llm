.PHONY: help dev test

help:
	@printf "Available targets:\n"
	@printf "  dev  Run gateway with uvicorn --reload using .env\n"
	@printf "  test Run gateway test suite\n"

dev:
	@set -a; \
	if [ -f .env ]; then . ./.env; fi; \
	set +a; \
	uv run --env-file .env uvicorn any_llm.gateway.dev:create_dev_app --factory --reload --host "$${GATEWAY_HOST:-0.0.0.0}" --port "$${GATEWAY_PORT:-8000}" --reload-dir src/any_llm/gateway

test:
	uv run pytest -v tests/gateway tests/unit/test_gateway_*.py
