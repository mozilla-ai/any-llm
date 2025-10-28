# Batch

!!! warning "Experimental API"
    The Batch API is experimental and subject to breaking changes in future versions. Use with caution in production environments.

The Batch API allows you to process multiple requests asynchronously at a lower cost.

## File Path Interface

The `any-llm` batch API requires you to pass a **path to a local JSONL file** containing your batch requests. The provider implementation automatically handles uploading and file management as needed.

Different providers handle batch processing differently:

- **OpenAI**: Requires uploading a file first, then creating a batch with the file ID
- **Anthropic** (future): Expects file content passed directly in the request
- **Other providers**: May have their own unique requirements

By accepting a local file path, `any-llm` abstracts these provider differences and handles the implementation details automatically.

::: any_llm.api.create_batch
::: any_llm.api.acreate_batch
::: any_llm.api.retrieve_batch
::: any_llm.api.aretrieve_batch
::: any_llm.api.cancel_batch
::: any_llm.api.acancel_batch
::: any_llm.api.list_batches
::: any_llm.api.alist_batches
