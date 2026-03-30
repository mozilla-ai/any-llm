from any_llm.gateway.auth.models import generate_api_key, hash_key, validate_api_key_format


def verify_api_key(*args, **kwargs):
    from any_llm.gateway.api.deps import verify_api_key as _verify_api_key

    return _verify_api_key(*args, **kwargs)


def verify_master_key(*args, **kwargs):
    from any_llm.gateway.api.deps import verify_master_key as _verify_master_key

    return _verify_master_key(*args, **kwargs)


def verify_api_key_or_master_key(*args, **kwargs):
    from any_llm.gateway.api.deps import verify_api_key_or_master_key as _verify_api_key_or_master_key

    return _verify_api_key_or_master_key(*args, **kwargs)


__all__ = [
    "generate_api_key",
    "hash_key",
    "validate_api_key_format",
    "verify_api_key",
    "verify_api_key_or_master_key",
    "verify_master_key",
]
