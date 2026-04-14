from any_llm.providers.openai.base import BaseOpenAIProvider


class DashscopeProvider(BaseOpenAIProvider):
    API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ENV_API_KEY_NAME = "DASHSCOPE_API_KEY"
    ENV_API_BASE_NAME = "DASHSCOPE_API_BASE"
    PROVIDER_NAME = "dashscope"
    PROVIDER_DOCUMENTATION_URL = "https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api"

    SUPPORTS_COMPLETION_REASONING = False
