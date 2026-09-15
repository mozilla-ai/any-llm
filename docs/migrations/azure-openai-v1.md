---
title: Azure OpenAI v1 migration
description: Migrate Azure OpenAI configuration to the v1 API
---

# Azure OpenAI v1 migration

The `azureopenai` provider now uses one OpenAI SDK client at `/openai/v1/`.
There is no fallback to the dated Azure API. Existing callers using dated
versions or `azure_deployment` must update their configuration.

## Update your configuration

Set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`, or supply the endpoint
and credential explicitly:

```python
from any_llm import AnyLLM

llm = AnyLLM.create(
    "azureopenai",
    api_base="https://YOUR-RESOURCE.openai.azure.com/",
    api_version="v1",  # Optional unless overriding a legacy environment value
)
response = llm.completion(
    model="YOUR-DEPLOYMENT-NAME",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

- Replace `api_version="2025-03-01-preview"` (or another dated version) with
  `"v1"`, or remove it and unset `OPENAI_API_VERSION`. Explicit `api_version`
  takes precedence over that environment variable. A selected dated version
  raises a migration error.
- Remove `azure_deployment`. Pass the deployment name as `model` on each request,
  including embeddings and media. The deployment is no longer part of the URL.
- Endpoint precedence is `api_base`, then `azure_endpoint`, then
  `AZURE_OPENAI_ENDPOINT`. Resource roots and URLs already ending in
  `/openai/v1/` are accepted. Custom proxy path prefixes are retained.
  Generic `OPENAI_BASE_URL` and `OPENAI_API_KEY` do not select an Azure resource
  or credential.
- Requests omit `api-version` by default. An explicit
  `default_query={"api-version": "v1"}` is accepted; other client-wide values
  are rejected even when `api_version="v1"`. Unrelated query entries, custom
  headers, HTTP clients, timeouts, and retry settings are preserved.

## Microsoft Entra authentication

Pass exactly one explicit credential: `api_key`, `azure_ad_token`, or
`azure_ad_token_provider`. Multiple explicit credentials raise an error.
When none is supplied, `AZURE_OPENAI_AD_TOKEN` takes precedence over
`AZURE_OPENAI_API_KEY`; empty environment values are treated as absent.
An explicit empty credential fails instead of falling back to the environment.

```python
from any_llm import AnyLLM
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

credential = DefaultAzureCredential()
llm = AnyLLM.create(
    "azureopenai",
    azure_endpoint="https://YOUR-RESOURCE.openai.azure.com/",
    azure_ad_token_provider=get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    ),
)
```

Install `azure-identity` separately to use this example. Synchronous token
providers run off the event loop; asynchronous providers are also accepted.
Providers must return a nonempty string. The SDK refreshes dynamic tokens on
requests and retries. Static tokens remain the caller's responsibility to refresh.
Close the SDK client with `await llm.client.close()` when finished, including when
you supply its HTTP client. Close credentials you created separately.

## Media on the same v1 client

Image generation, audio transcription, and speech retain their capability flags.
Their requests use `/openai/v1/images/generations`,
`/openai/v1/audio/transcriptions`, and `/openai/v1/audio/speech`, respectively.
These operations automatically select `api-version=preview`, overriding a
client-wide `api-version=v1` only for media. Chat, Responses, embeddings, and
model listing do not switch to preview.

Each operation needs a compatible deployment and region. An advertised
capability does not mean your resource has that deployment. Select the model
version when deploying in Azure; pass the deployment name on requests.

See Microsoft's [v1 migration guide](https://learn.microsoft.com/azure/foundry/openai/api-version-lifecycle)
and [v1 preview media reference](https://learn.microsoft.com/azure/ai-foundry/openai/reference-preview-latest).
