import logging

from any_llm.types.completion import CompletionParams

logger = logging.getLogger(__name__)


def resolve_output_token_limit(params: CompletionParams) -> int | None:
    """Return the output token limit for providers that name it themselves.

    ``max_completion_tokens`` is the OpenAI name for the limit and supersedes ``max_tokens``, so it
    wins when both are set, as it does for the OpenAI-compatible providers.
    """
    if params.max_completion_tokens is None:
        return params.max_tokens
    if params.max_tokens is not None and params.max_tokens != params.max_completion_tokens:
        logger.warning(
            "Ignoring max_tokens (%s) in favor of max_completion_tokens (%s).",
            params.max_tokens,
            params.max_completion_tokens,
        )
    return params.max_completion_tokens
