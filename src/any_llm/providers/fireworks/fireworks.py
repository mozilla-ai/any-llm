from any_llm.providers.openai.base import BaseOpenAIProvider


class FireworksProvider(BaseOpenAIProvider):
    PROVIDER_NAME = "fireworks"
    ENV_API_KEY_NAME = "FIREWORKS_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://fireworks.ai/api"
    API_BASE = "https://api.fireworks.ai/inference/v1"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True

    SUPPORTS_RESPONSES = False
    # Fireworks is not compliant with OpenResponses spec
    """
           if isinstance(response, OpenAIResponse):
>           return ResponseResource.model_validate(response.model_dump())
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           pydantic_core._pydantic_core.ValidationError: 45 validation errors for ResponseResource
E           output.1.Message.content.0.InputTextContent.type
E             Input should be 'input_text' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.Message.content.0.OutputTextContent.annotations
E             Input should be a valid list [type=list_type, input_value=None, input_type=NoneType]
E               For further information visit https://errors.pydantic.dev/2.12/v/list_type
E           output.1.Message.content.0.OutputTextContent.logprobs
E             Input should be a valid list [type=list_type, input_value=None, input_type=NoneType]
E               For further information visit https://errors.pydantic.dev/2.12/v/list_type
E           output.1.Message.content.0.TextContent.type
E             Input should be 'text' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.Message.content.0.SummaryTextContent.type
E             Input should be 'summary_text' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.Message.content.0.ReasoningTextContent.type
E             Input should be 'reasoning_text' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.Message.content.0.RefusalContent.type
E             Input should be 'refusal' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.Message.content.0.RefusalContent.refusal
E             Field required [type=missing, input_value={'annotations': None, 'te...text', 'logprobs': None}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.Message.content.0.InputImageContent.type
E             Input should be 'input_image' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.Message.content.0.InputImageContent.image_url
E             Field required [type=missing, input_value={'annotations': None, 'te...text', 'logprobs': None}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.Message.content.0.InputImageContent.detail
E             Field required [type=missing, input_value={'annotations': None, 'te...text', 'logprobs': None}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.Message.content.0.InputFileContent.type
E             Input should be 'input_file' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.Message.content.0.InputVideoContent.type
E             Input should be 'input_video' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.Message.content.0.InputVideoContent.video_url
E             Field required [type=missing, input_value={'annotations': None, 'te...text', 'logprobs': None}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.FunctionCall.type
E             Input should be 'function_call' [type=literal_error, input_value='message', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.FunctionCall.call_id
E             Field required [type=missing, input_value={'id': 'msg_0d165df84a344...ted', 'type': 'message'}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.FunctionCall.name
E             Field required [type=missing, input_value={'id': 'msg_0d165df84a344...ted', 'type': 'message'}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.FunctionCall.arguments
E             Field required [type=missing, input_value={'id': 'msg_0d165df84a344...ted', 'type': 'message'}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.FunctionCallOutput.type
E             Input should be 'function_call_output' [type=literal_error, input_value='message', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.FunctionCallOutput.call_id
E             Field required [type=missing, input_value={'id': 'msg_0d165df84a344...ted', 'type': 'message'}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.FunctionCallOutput.output
E             Field required [type=missing, input_value={'id': 'msg_0d165df84a344...ted', 'type': 'message'}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.ReasoningBody.type
E             Input should be 'reasoning' [type=literal_error, input_value='message', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.ReasoningBody.content.0.InputTextContent.type
E             Input should be 'input_text' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.ReasoningBody.content.0.OutputTextContent.annotations
E             Input should be a valid list [type=list_type, input_value=None, input_type=NoneType]
E               For further information visit https://errors.pydantic.dev/2.12/v/list_type
E           output.1.ReasoningBody.content.0.OutputTextContent.logprobs
E             Input should be a valid list [type=list_type, input_value=None, input_type=NoneType]
E               For further information visit https://errors.pydantic.dev/2.12/v/list_type
E           output.1.ReasoningBody.content.0.TextContent.type
E             Input should be 'text' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.ReasoningBody.content.0.SummaryTextContent.type
E             Input should be 'summary_text' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.ReasoningBody.content.0.ReasoningTextContent.type
E             Input should be 'reasoning_text' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.ReasoningBody.content.0.RefusalContent.type
E             Input should be 'refusal' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.ReasoningBody.content.0.RefusalContent.refusal
E             Field required [type=missing, input_value={'annotations': None, 'te...text', 'logprobs': None}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.ReasoningBody.content.0.InputImageContent.type
E             Input should be 'input_image' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.ReasoningBody.content.0.InputImageContent.image_url
E             Field required [type=missing, input_value={'annotations': None, 'te...text', 'logprobs': None}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.ReasoningBody.content.0.InputImageContent.detail
E             Field required [type=missing, input_value={'annotations': None, 'te...text', 'logprobs': None}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           output.1.ReasoningBody.content.0.InputFileContent.type
E             Input should be 'input_file' [type=literal_error, input_value='output_text', input_type=str]
E               For further information visit https://errors.pydantic.dev/2.12/v/literal_error
E           output.1.ReasoningBody.summary
E             Field required [type=missing, input_value={'id': 'msg_0d165df84a344...ted', 'type': 'message'}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           text
E             Input should be a valid dictionary or instance of TextField [type=model_type, input_value=None, input_type=NoneType]
E               For further information visit https://errors.pydantic.dev/2.12/v/model_type
E           presence_penalty
E             Field required [type=missing, input_value={'id': 'resp_b2f8b3fa508c...r': None, 'store': True}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           frequency_penalty
E             Field required [type=missing, input_value={'id': 'resp_b2f8b3fa508c...r': None, 'store': True}, input_type=dict]
E               For further information visit https://errors.pydantic.dev/2.12/v/missing
E           top_logprobs
E             Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]
E               For further information visit https://errors.pydantic.dev/2.12/v/int_type
E           usage.input_tokens
E             Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]
E               For further information visit https://errors.pydantic.dev/2.12/v/int_type
E           usage.output_tokens
E             Input should be a valid integer [type=int_type, input_value=None, input_type=NoneType]
E               For further information visit https://errors.pydantic.dev/2.12/v/int_type
E           usage.input_tokens_details
E             Input should be a valid dictionary or instance of InputTokensDetails [type=model_type, input_value=None, input_type=NoneType]
E               For further information visit https://errors.pydantic.dev/2.12/v/model_type
E           usage.output_tokens_details
E             Input should be a valid dictionary or instance of OutputTokensDetails [type=model_type, input_value=None, input_type=NoneType]
E               For further information visit https://errors.pydantic.dev/2.12/v/model_type
E           background
E             Input should be a valid boolean [type=bool_type, input_value=None, input_type=NoneType]
E               For further information visit https://errors.pydantic.dev/2.12/v/bool_type
E           service_tier
E             Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]
E               For further information visit https://errors.pydantic.dev/2.12/v/string_type

src/any_llm/providers/openai/base.py:172: ValidationError
"""
