## OpenResponses API

The Responses API in any-llm implements the [OpenResponses](https://www.openresponses.org/) specification—an open-source standard for building multi-provider, interoperable LLM interfaces for agentic AI systems.

!!! info "Learn More"

    - [OpenResponses Specification](https://www.openresponses.org/specification)
    - [OpenResponses Reference](https://www.openresponses.org/reference)
    - [HuggingFace Responses API Guide](https://huggingface.co/docs/inference-providers/guides/responses-api)

### Return Types

The `responses()` and `aresponses()` functions return different types depending on the provider's level of OpenResponses compliance:

| Return Type | When Returned |
|-------------|---------------|
| `openresponses_types.ResponseResource` | Providers fully compliant with the OpenResponses specification |
| `openai.types.responses.Response` | Providers using OpenAI's native Responses API (not yet fully OpenResponses-compliant) |
| `Iterator[dict]` / `AsyncIterator[dict]` | When `stream=True` is set |

!!! tip "Handling Multiple Return Types"

    Since the return type varies by provider, you may need to handle both cases:

    ```python
    from any_llm import responses
    from openresponses_types import ResponseResource
    from openai.types.responses import Response

    result = responses("openai:gpt-4o", "Hello, world!")

    if isinstance(result, ResponseResource):
        # OpenResponses-compliant provider
        print(result.output)
    elif isinstance(result, Response):
        # OpenAI-native provider
        print(result.output)
    ```

    Both `ResponseResource` and `Response` share a similar structure, so in many cases
    you can access common fields like `output` without type checking.

::: any_llm.api.responses
::: any_llm.api.aresponses
