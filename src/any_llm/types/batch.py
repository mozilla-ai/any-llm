from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openai.types import Batch as OpenAIBatch
from openai.types.batch_request_counts import BatchRequestCounts as OpenAIBatchRequestCounts

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion

# Right now it's a direct copy but I'll re-export them here,
#  so that if we need to expand them in the future, people won't need to update their imports
Batch = OpenAIBatch
BatchRequestCounts = OpenAIBatchRequestCounts


@dataclass
class BatchResultError:
    """An error that occurred for a single request within a batch."""

    code: str
    message: str


@dataclass
class BatchResultItem:
    """The result of a single request within a batch."""

    custom_id: str
    result: ChatCompletion | None = None
    error: BatchResultError | None = None


@dataclass
class BatchResult:
    """The results of a completed batch job."""

    results: list[BatchResultItem] = field(default_factory=list)
