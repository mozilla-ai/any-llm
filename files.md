---
title: Files
description: Manage provider-hosted files through an AnyLLM instance
---

# Files

The Files API exposes upload, listing, metadata retrieval, streamed download, and
deletion on an `AnyLLM` instance. Anthropic, OpenAI, and Azure OpenAI support all
five operations. Other providers, including custom OpenAI-compatible endpoints,
do not automatically inherit Files support.

| Provider | Upload | List | Retrieve | Download | Delete |
| --- | --- | --- | --- | --- | --- |
| Anthropic | Yes | Yes | Yes | Generated files | Yes |
| OpenAI | Yes, requires `purpose` | Yes | Yes | Depends on file purpose | Yes |
| Azure OpenAI (v1) | Yes, requires `purpose` | Yes | Yes | Depends on file purpose | Yes |

Check `get_provider_metadata().file_operations` before using Files on a provider.

```python
from any_llm import AnyLLM

provider = AnyLLM.create("anthropic")  # ANTHROPIC_API_KEY
capabilities = provider.get_provider_metadata()
print(capabilities.files)
print(capabilities.file_operations)
```

## Upload and use a file

```python
from pathlib import Path
from any_llm import AnyLLM

provider = AnyLLM.create("anthropic")
uploaded = provider.upload_file(
    Path("report.pdf"),
    mime_type="application/pdf",
    expires_in=3600,
)
try:
    response = provider.messages(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "file", "file_id": uploaded.id}},
                {"type": "text", "text": "Summarize this report."},
            ],
        }],
    )
    print(response.content)
finally:
    provider.delete_file(uploaded.id)
```

`file` accepts a path (`str` or `PathLike`), `bytes`, or a binary file handle.
`filename` and `mime_type` override the multipart part's metadata. The default
filename is the path's basename, or `upload` for bytes and handles; the default
MIME type is `application/octet-stream`. Supply the actual MIME type for inputs
whose interpretation depends on it.

Path uploads open a binary handle and close it after the request. Caller-owned
handles remain open. The SDK multipart encoder reads handles in bounded chunks;
any-llm does not read the entire path into bytes first. These file reads are
synchronous SDK I/O even for an asynchronous upload. Bytes inputs are already
resident in memory. Async iterators are not supported upload inputs.

## OpenAI uploads

OpenAI requires an explicit, nonempty `purpose` for every upload. Choose the
purpose for the API that will consume the file: `user_data` for general model
inputs, `batch` for Batch API input JSONL, or `fine-tune` for training data.
Other upload purposes include `assistants`, `vision`, and `evals`. OpenAI validates
accepted purposes, file formats, and account-specific limits on the server.

```python
from any_llm import AnyLLM

openai_provider = AnyLLM.create("openai")  # OPENAI_API_KEY
uploaded = openai_provider.upload_file(
    b"Revenue grew by 10 percent.\n",
    filename="report.txt",
    mime_type="text/plain",
    purpose="user_data",
    expires_in=3600,
)
try:
    metadata = openai_provider.retrieve_file(uploaded.id)
    print(metadata.id, metadata.size_bytes, metadata.expires_at)
finally:
    openai_provider.delete_file(uploaded.id)
```

`expires_in` maps to OpenAI's `expires_after` with `anchor="created_at"` and
`seconds=expires_in`. Omit it to use the provider's retention policy. The server
validates supported durations and purpose combinations. Native `expires_after`
keyword arguments are not accepted; use the shared `expires_in` parameter.

OpenAI's `bytes` becomes `size_bytes`, and timestamps become `datetime` values.
`purpose` and `status` are preserved when returned. Missing metadata, including
`mime_type` and `downloadable`, remains `None`; do not interpret an unknown
`downloadable` value as permission to download.

## Azure OpenAI Files

Azure OpenAI uses the same public methods and metadata normalization as OpenAI,
but routes requests to your Azure resource's `/openai/v1/files` endpoint. Configure
`AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`, or use the provider's existing
Microsoft Entra authentication options (`azure_ad_token` or
`azure_ad_token_provider`). Files requests use the v1 routes without adding a
preview query parameter or a deployment name to the URL.

```python
from any_llm import AnyLLM

azure_provider = AnyLLM.create("azureopenai")
page = azure_provider.list_files(limit=20, purpose="batch")
print(page.next_cursor)
```

The [Azure Files v1 reference](https://learn.microsoft.com/en-us/rest/api/microsoft-foundry/azureopenai/files)
lists upload purposes `assistants`, `batch`, `fine-tune`, and `evals`. Do not assume
that OpenAI's `user_data` or `vision` upload purposes are available on Azure.
Azure validates accepted purposes, file formats, expiry durations, and download
permissions on the server. `expires_in` uses the same `expires_after` mapping as
OpenAI; the adapter does not invent an expiry when it is omitted. For batch
uploads, [Microsoft documents](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/batch)
1,209,600 to 2,592,000 seconds (14 to 30 days). The Azure resource used by our
integration tests also accepted 259,200 seconds (3 days), but this does not
establish a universal minimum. Use 14 to 30 days unless you have confirmed a
lower minimum on your resource. A one-hour expiry accepted by OpenAI was
rejected by our Azure resource with HTTP 400. Verify the retention behavior
for the purpose and Azure resource you use.

Azure shares the `cursor`/`next_cursor` pagination contract and supports the
`purpose` filter and `order` option. File IDs remain scoped to their originating
Azure resource and account; do not pass OpenAI file IDs to Azure or vice versa.

## Methods

| Synchronous | Asynchronous | Result |
| --- | --- | --- |
| `upload_file(file, filename=None, mime_type=None, purpose=None, expires_in=None, **kwargs)` | `await aupload_file(...)` | `FileMetadata` |
| `list_files(limit=None, cursor=None, purpose=None, **kwargs)` | `await alist_files(...)` | `FilePage` |
| `retrieve_file(file_id, **kwargs)` | `await aretrieve_file(...)` | `FileMetadata` |
| `download_file(file_id, chunk_size=65536, **kwargs)` | `adownload_file(...)` | Context manager yielding `FileDownload` / `AsyncFileDownload` |
| `delete_file(file_id, **kwargs)` | `await adelete_file(...)` | `FileDeleted` |

These are instance methods. Reuse the instance configured for the originating
provider account; file IDs are not portable to another provider or account.
Use async methods inside an event loop. Sync methods honor the existing
`allow_running_loop` option.

`FileMetadata`, `FilePage`, and `FileDeleted` are exported from `any_llm` and
`any_llm.types.files`. Metadata uses `size_bytes`, `mime_type`, and parsed
`datetime` timestamps, plus `purpose`, `status`, and `downloadable`.
Fields a provider omits remain `None`. Each provider normalizes its own responses. Provider-specific
fields survive in `model_extra` and `model_dump()`. A deletion acknowledgement
preserves its native fields, without inventing a `deleted` flag if absent.

## Pagination

Listing fetches exactly one page. Use the opaque `next_cursor` with the same
provider, account, and filters to continue; `None` marks the final page.

```python
page = provider.list_files(limit=20)
if page.next_cursor is not None:
    next_page = provider.list_files(limit=20, cursor=page.next_cursor)
```

Providers translate their native pagination into `cursor` and `next_cursor`.
OpenAI and Azure OpenAI map `cursor` to `after` and derive `next_cursor` from the last file ID
when `has_more` is true. Use `purpose="batch"` to filter a listing and
`order="asc"` or `order="desc"` to select the order. Keep the same purpose and
order on subsequent pages. Do not pass the native `after` keyword.

Anthropic maps `cursor` and `next_cursor` to `page` and `next_page` internally.
Native cursor keyword arguments are not supported, and listing rejects the legacy
`files-api-2025-04-14` beta because it changes the page shape. Other Files
operations forward that beta unchanged.

For a known set of IDs, Anthropic accepts the provider-specific `ids=[...]`
option. The server validates its limits and combinations with other parameters.
Missing or inaccessible IDs are omitted.

## Downloads and provider restrictions

### OpenAI

Download eligibility depends on the file's purpose. Batch input files can be
downloaded through the Files API without submitting a batch job. OpenAI rejects
downloads of `user_data` files with HTTP 400, even when uploading and retrieving
their metadata succeeds. Other purposes have their own restrictions; consult
[OpenAI's Files reference](https://platform.openai.com/docs/api-reference/files)
before assuming a file can be downloaded.

For a downloadable file ID, use `openai_provider.download_file(file_id)` or
`openai_provider.adownload_file(file_id)` with the context-manager patterns below.
OpenAI's deletion acknowledgement preserves its native `deleted` flag and
provider-specific fields such as `object`.

### Azure OpenAI

Azure downloads use `/openai/v1/files/{file_id}/content` with the same streaming
context managers. Download eligibility is determined by Azure, not by the OpenAI
purpose restrictions above. A failure on context entry is propagated through the
normal error mapping; it is not treated as an empty download.

### Anthropic

Anthropic marks user uploads as non-downloadable. Download files returned by
native code execution or supported skills, using their structured `file_id`.
For example, after a Messages request using the native code execution tool:

```python
response = provider.messages(
    model="claude-sonnet-4-6",
    max_tokens=2048,
    tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
    messages=[{"role": "user", "content": "Create a small CSV in /mnt/data and return it as a downloadable file."}],
    timeout=90,
)
for block in response.content:
    if block.type == "bash_code_execution_tool_result":
        for output in block.content.content:
            if output.type == "bash_code_execution_output":
                metadata = provider.retrieve_file(output.file_id)
                if metadata.downloadable:
                    with provider.download_file(output.file_id) as chunks:
                        with open("result.bin", "wb") as destination:
                            for chunk in chunks:
                                destination.write(chunk)
```

For an async caller with a generated file ID:

```python
async def download_to_writer(provider, file_id, consume):
    async with provider.adownload_file(file_id, chunk_size=65536, timeout=30) as download:
        content_type = download.headers.get("content-type", "application/octet-stream")
        async for chunk in download:
            await consume(chunk)  # Your async writer
```

The request starts on entering `with` or `async with`. Connection failures and
HTTP errors are raised before the context body runs, so a gateway can return the
correct error status before committing its downstream response.

The public `FileDownload` and `AsyncFileDownload` types expose `status_code` and
`headers` as soon as the context is entered. Response header lookup is
case-insensitive for Anthropic, OpenAI, and Azure OpenAI. Both objects remain directly
iterable over bytes, so existing
`for chunk in download` and `async for chunk in download` loops keep working.
They are exported from `any_llm` and `any_llm.types.files`.

Body reads begin only on iteration. Always use the context manager, even if you
only inspect headers: exiting closes the upstream response after full
consumption, early exit, or cancellation. The sync bridge requests one chunk at
a time without prefetching the body. Exceptions raised by consumer code are not
converted into provider errors.

Headers describe the upstream response, while body chunks use the transport's
content decoding. A proxy must choose an explicit response-header allowlist;
it must not blindly forward `Content-Encoding` or a compressed `Content-Length`
for decoded bytes. Header values can include sensitive metadata such as filenames
and should not be logged indiscriminately.

## Provider options and errors

All OpenAI and Azure OpenAI Files methods accept `timeout`, `max_retries`, and
`extra_headers`. Upload requires the shared `purpose` parameter and accepts
`expires_in`. Listing accepts `purpose` and the provider-specific `order` option.
Anthropic's `betas` and `ids` options are not accepted by either provider.

All Anthropic Files methods accept `timeout`, `max_retries`, `betas`, and
`extra_headers` (for example, `{"anthropic-version": "2023-06-01"}`). Upload also
accepts the shared `expires_in` parameter, a positive integer duration in seconds,
which maps to Anthropic's `expires_in_seconds`. Omission leaves retention to the
provider. The shared `purpose` parameter is available for upload and listing;
Anthropic rejects non-`None` values. Provider-specific options remain in `**kwargs`.
Beta values from configured headers, request headers, and `betas` are merged.
Anthropic SDK 0.124.0 or newer is required.

Unsupported options raise `UnsupportedParameterError`. Nonpositive `limit`,
invalid `expires_in`, nonpositive `chunk_size`, invalid file IDs, and upload
paths that cannot be opened raise `InvalidRequestError`. Local file-opening
errors are preserved in `original_exception`, regardless of the unified-exceptions setting.
Provider-specific numeric limits and option combinations are validated by the server, so SDK
updates do not require copying server limits into any-llm.

Uploads default to **zero automatic retries**, even if the provider instance
has retries enabled. A caller can explicitly override `max_retries`, but a lost
response can follow successful creation: retrying can create another file.
Other operations inherit the configured SDK retry policy unless overridden.
Timeouts do not prove that an upload failed before creation.

Errors follow the existing `ANY_LLM_UNIFIED_EXCEPTIONS` setting. When enabled,
a retrieve, download, or delete HTTP 404 becomes `ProviderFileNotFoundError`, authentication errors
remain `AuthenticationError`, and `RateLimitError` retains `retry_after`.
Upload and list 404s retain the existing general error mapping.
Streaming failures are converted during iteration. Without unified errors,
SDK exceptions retain their existing behavior.

Applications remain responsible for authorization, ownership, cleanup jobs,
credential rotation, and provider-account selection. This API neither stores
application copies nor makes provider files shareable between tenants.
