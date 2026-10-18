from any_llm import AnyLLM


def test_nebius_supports_responses() -> None:
    provider = AnyLLM.create("nebius", api_key="dummy_key")
    assert provider.SUPPORTS_RESPONSES is True
