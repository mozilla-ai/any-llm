from typing import TypeAlias

from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ResponseInputParam as OpenAIResponseInputParam
from openai.types.responses import ResponseOutputMessage as OpenAIResponseOutputMessage
from openai.types.responses import ResponseStreamEvent as OpenAIResponseStreamEvent

Response: TypeAlias = OpenAIResponse
ResponseStreamEvent: TypeAlias = OpenAIResponseStreamEvent
ResponseOutputMessage: TypeAlias = OpenAIResponseOutputMessage
ResponseInputParam: TypeAlias = OpenAIResponseInputParam
