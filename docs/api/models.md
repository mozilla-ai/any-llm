# Fetching Available Models

AnyLLM providers that support model listing expose a `models()` method, which returns metadata for each available model.

## Model Metadata

Each model is described by a `ModelMetadata` object:

```python
class ModelMetadata(BaseModel):
    id: str
    created: int | None = None
    object: str | None = None
```

- `id`: Unique model identifier (e.g., "openai/gpt-4o")
- `created`: Optional creation timestamp
- `object`: Optional type descriptor

## How to Fetch Models

### Synchronous Example

```python
from any_llm.provider import ProviderFactory, ApiConfig

provider = ProviderFactory.create_provider("openai", ApiConfig(api_key="YOUR_API_KEY"))
models = provider.models()
for model in models:
    print(model.id)
```

### Asynchronous Example

```python
import asyncio
from any_llm.provider import ProviderFactory, ApiConfig

async def fetch_models():
    provider = ProviderFactory.create_provider("openai", ApiConfig(api_key="YOUR_API_KEY"))
    models = await provider.amodels()
    for model in models:
        print(model.id)

asyncio.run(fetch_models())
```

## Notes

- Not all providers support model listing. Check `provider.SUPPORTS_LIST_MODELS`.
- Returned models may include additional metadata depending on the provider.
- Use the model `id` when making API calls.
