import pytest
from unittest.mock import Mock

from any_llm.providers.gemini import utils

# 1. _convert_tool_spec


def test_convert_tool_spec_valid():
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "test_func",
                "description": "desc",
                "parameters": {
                    "properties": {
                        "param1": {"type": "string", "description": "desc1"},
                        "param2": {"type": "array", "description": "desc2", "items": {"type": "string"}},
                    },
                    "required": ["param1"],
                },
            },
        }
    ]
    result = utils._convert_tool_spec(openai_tools)
    assert result


def test_convert_tool_spec_non_function():
    openai_tools = [{"type": "not_function"}]
    result = utils._convert_tool_spec(openai_tools)
    assert result == [utils.types.Tool(function_declarations=[])]


# 2. _convert_tool_choice


def test_convert_tool_choice_required():
    result = utils._convert_tool_choice("required")
    assert result


def test_convert_tool_choice_auto():
    result = utils._convert_tool_choice("auto")
    assert result


@pytest.mark.parametrize("choice", ["invalid", "", None])
def test_convert_tool_choice_invalid(choice):
    with pytest.raises(KeyError):
        utils._convert_tool_choice(choice)


# 3. _convert_messages


def test_convert_messages_system_and_user():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    formatted, system_instruction = utils._convert_messages(messages)
    assert system_instruction == "sys"
    assert formatted


def test_convert_messages_user_image_base64():
    import base64

    valid_png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    messages = [{"role": "user", "content": "img", "image_base64": valid_png}]
    formatted, _ = utils._convert_messages(messages)
    assert formatted


def test_convert_messages_user_image_base64_invalid():
    messages = [{"role": "user", "content": "img", "image_base64": "not_base64!"}]
    with pytest.raises(ValueError):
        utils._convert_messages(messages)


def test_convert_messages_user_image_url():
    messages = [{"role": "user", "content": "img", "image_url": "https://valid-url.com/img.png"}]
    formatted, _ = utils._convert_messages(messages)
    assert formatted


def test_convert_messages_user_image_url_invalid():
    messages = [{"role": "user", "content": "img", "image_url": "ftp://invalid-url"}]
    with pytest.raises(ValueError):
        utils._convert_messages(messages)


def test_convert_messages_user_image_bytes():
    messages = [{"role": "user", "content": "img", "image_bytes": b"bytes"}]
    formatted, _ = utils._convert_messages(messages)
    assert formatted


def test_convert_messages_user_image_bytes_invalid():
    messages = [{"role": "user", "content": "img", "image_bytes": "not_bytes"}]
    with pytest.raises(ValueError):
        utils._convert_messages(messages)


def test_convert_messages_assistant_tool_calls():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "func", "arguments": "{}"}}],
        }
    ]
    formatted, _ = utils._convert_messages(messages)
    assert formatted


def test_convert_messages_tool_valid_json():
    messages = [{"role": "tool", "content": '{"result": 1}', "name": "toolname"}]
    formatted, _ = utils._convert_messages(messages)
    assert formatted


def test_convert_messages_tool_invalid_json():
    messages = [{"role": "tool", "content": "not_json", "name": "toolname"}]
    formatted, _ = utils._convert_messages(messages)
    assert formatted


# ...additional tests for other functions can be added similarly...
