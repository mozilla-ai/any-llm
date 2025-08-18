from any_llm.providers.openai.base import BaseOpenAIProvider


class DatabricksProvider(BaseOpenAIProvider):
    """Databricks Provider using the new response conversion utilities."""

    ENV_API_KEY_NAME = "DATABRICKS_TOKEN"
    PROVIDER_NAME = "databricks"
    PROVIDER_DOCUMENTATION_URL = "https://docs.databricks.com/"
