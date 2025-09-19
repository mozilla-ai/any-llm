from any_llm.providers.llamacpp.llamacpp import LlamacppProvider


def test_provider_without_api_key() -> None:
    provider = LlamacppProvider()
    assert provider.PROVIDER_NAME == "llamacpp"
