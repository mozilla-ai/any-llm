# AnyLLM

All providers must inherit from this class and implement the following methods:

- Provider initialization

Public method to get a provider-specific class.

```py
@classmethod
async def acreate(
    cls, provider: str | LLMProvider, api_key: str | None = None, api_base: str | None = None, **kwargs: Any
) -> AnyLLM:
```

Providers must be loaded dynamically with lazy imports to allow users
to only install and import code from the providers they need.

- Client initialization

Internal method setting the HTTP / Provider SDK client to be reused across method calls.
 
When api_key is None, this method should check for the standard environment variables
that each provider might use (for example, OPENAI_API_KEY).

```py
@abstractmethod
def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
```

- Completion call

See [completions.md](./completions.md).

All the params must be exposed as arguments to this function.
Additional arguments are supported to be passed to providers that might accept them.

```py
@abstractmethod
async def acompletion(self, model: str, ... , **kwargs: Any) -> ChatCompletion:
    pass

@abstractmethod
async def acompletion_stream(self, model: str, ... ,  **kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
    """Perform a request following completions.md using stream=True
    Should return an iterator if the language supports it.
    """
    pass
```

### Internal methods

For consistency in providers with their own SDKS, the following set
of helper methods should be implemented to encapsulate the conversions
of provider-specific inputs/outputs to the OpenAI "standard":

```py
@staticmethod
@abstractmethod
def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
    pass

@staticmethod
@abstractmethod
def _convert_completion_response(response: Any) -> ChatCompletion:
    pass

@staticmethod
@abstractmethod
def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
    pass
```
