## OpenResponses Types

Data models and types for the [OpenResponses](https://www.openresponses.org/) API specification.

### Response Types

Response types are imported directly from the OpenAI SDK:

```python
from openai.types.responses import Response, ResponseStreamEvent
```

### Parameter Types

Parameter types for building requests are available from `openresponses_types`:

```python
from openresponses_types import ResponsesParams, ResponseInputParam
```

For the full OpenResponses type definitions, see the [openresponses-types](https://pypi.org/project/openresponses-types/) package documentation.
