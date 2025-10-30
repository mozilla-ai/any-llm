from openai.types import Batch as OpenAIBatch
from openai.types.batch_request_counts import BatchRequestCounts as OpenAIBatchRequestCounts

# Right now it's a direct copy but I'll re-export them here,
#  so that if we need to expand them in the future, people won't need to update their imports
Batch = OpenAIBatch
BatchRequestCounts = OpenAIBatchRequestCounts
