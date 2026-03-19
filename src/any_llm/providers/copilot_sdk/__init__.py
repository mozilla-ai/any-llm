from .copilot_sdk import CopilotSdkProvider

# Factory alias: AnyLLM._create_provider() derives class names via
# provider_key.capitalize() + "Provider", which yields "Copilot_sdkProvider".
Copilot_sdkProvider = CopilotSdkProvider

__all__ = ["CopilotSdkProvider", "Copilot_sdkProvider"]
