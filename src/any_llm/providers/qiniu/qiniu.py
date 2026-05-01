from any_llm.providers.openai.base import BaseOpenAIProvider


class QiniuProvider(BaseOpenAIProvider):
    API_BASE = "https://api.qnaigc.com/v1"
    ENV_API_KEY_NAME = "QINIU_API_KEY"
    ENV_API_BASE_NAME = "QINIU_API_BASE"
    PROVIDER_NAME = "qiniu"
    PROVIDER_DOCUMENTATION_URL = "https://developer.qiniu.com/aitokenapi"

    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_COMPLETION_REASONING = True
