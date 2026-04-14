from any_llm.providers.openai.base import BaseOpenAIProvider


class AliyunProvider(BaseOpenAIProvider):
    API_BASE = "https://dashscope.aliyuncs.com/compatible-mode"
    ENV_API_KEY_NAME = "ALIYUN_API_KEY"
    ENV_API_BASE_NAME = "ALIYUN_API_BASE"
    PROVIDER_NAME = "aliyun"
    PROVIDER_DOCUMENTATION_URL = "https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api"

    SUPPORTS_COMPLETION_REASONING = False

    # Add any provider-specific parameter conversions if needed
    # @staticmethod
    # @override
    # def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
    #     converted_params = BaseOpenAIProvider._convert_completion_params(params, **kwargs)
    #     # Handle any parameter renaming or transformations
    #     return converted_params
