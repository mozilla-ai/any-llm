## OpenResponses API

The Responses API in any-llm implements the [OpenResponses](https://www.openresponses.org/) specification—an open-source standard for building multi-provider, interoperable LLM interfaces for agentic AI systems.

OpenResponses extends OpenAI's Responses API with:

- **Reasoning content**: Raw, encrypted, or summarized reasoning traces
- **Semantic streaming**: Events like `response.output_text.delta` instead of raw deltas
- **MCP tool support**: [Model Context Protocol](https://modelcontextprotocol.io/) server integration
- **Provider routing**: Use `model:provider` syntax to route requests (e.g., `moonshotai/Kimi-K2-Instruct:groq`)

### Supported Providers

The following providers support the OpenResponses API:

| Provider | Notes |
|----------|-------|
| `openai` | Native OpenAI Responses API |
| `azureopenai` | Azure OpenAI Service |
| `huggingface` | Via [HuggingFace OpenResponses Router](https://huggingface.co/docs/inference-providers/guides/responses-api) |
| `groq` | Direct Groq API |
| `fireworks` | Fireworks AI |

### Basic Usage

```python
from any_llm import AnyLLM, LLMProvider

# Using HuggingFace OpenResponses router
llm = AnyLLM.create(LLMProvider.HUGGINGFACE)

response = await llm.aresponses(
    model="openai/gpt-oss-120b:groq",  # model:provider routing
    input_data="What is the capital of France?",
    instructions="Be concise.",
)

print(response.output_text)
```

### With Reasoning

```python
from any_llm.types.responses import ReasoningEffort

response = await llm.aresponses(
    model="openai/gpt-oss-120b:groq",
    input_data="What is 15% of 80?",
    reasoning={"effort": "high"},  # or ReasoningEffort.HIGH
)

print(f"Answer: {response.output_text}")
print(f"Reasoning: {response.reasoning}")
```

### With MCP Tools

```python
from any_llm.types.responses import MCPTool

response = await llm.aresponses(
    model="moonshotai/Kimi-K2-Instruct:groq",
    input_data="Search for the latest news about AI",
    tools=[
        MCPTool(
            server_label="search",
            server_url="https://mcp.example.com/search",
            allowed_tools=["web_search"],
        )
    ],
)
```

### Streaming

```python
stream = await llm.aresponses(
    model="openai/gpt-oss-120b:groq",
    input_data="Tell me a story",
    stream=True,
)

async for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
```

!!! info "Learn More"

    - [OpenResponses Specification](https://www.openresponses.org/specification)
    - [OpenResponses Reference](https://www.openresponses.org/reference)
    - [HuggingFace Responses API Guide](https://huggingface.co/docs/inference-providers/guides/responses-api)

::: any_llm.api.responses
::: any_llm.api.aresponses
