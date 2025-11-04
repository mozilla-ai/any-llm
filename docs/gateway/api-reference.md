# API Reference

## OpenAPI Specification

The complete OpenAPI specification for the any-llm Gateway API is available at [`openapi.json`](openapi.json).

To regenerate the OpenAPI spec after making changes to the API routes, run:
```bash
python scripts/generate_openapi.py
```

## Chat Completions

All chat completion endpoints require the use of a client key: these are keys created by someone using the master key.

Look for the routes in `src/any_llm/gateway/chat.py`

## Key Management

All key management endpoints require the master key in the `X-AnyLLM-Key` header.

Look for the routes in `src/any_llm/gateway/keys.py`

## User Management

User management endpoints allow you to create, update, and manage users and their associated budgets.

## Budget Management

Budget management endpoints allow you to create, update, and manage spending limits for users.

See [Budget Management](budget-management.md) for detailed examples.

## Pricing Management

### Configuration-based Pricing

Pricing can be configured in the config file. See [Configuration](configuration.md#model-pricing-configuration) for details.

When both config and database pricing exist for a model, database pricing takes precedence.

Pricing management endpoints allow you to configure token pricing for different models. All pricing endpoints are located under `/v1/pricing`.
