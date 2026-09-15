---
title: Files
description: Manage provider-hosted files through an AnyLLM instance
---

# Files

The Files API exposes upload, listing, metadata retrieval, streamed download, and
deletion on an `AnyLLM` instance. Anthropic is the first supported provider.
Other providers, including OpenAI-compatible endpoints, do not automatically
inherit Files support.

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
Anthropic maps these to `page` and `next_page` internally. Native cursor
keyword arguments are not supported, and listing rejects the legacy
`files-api-2025-04-14` beta because it changes the page shape. Other Files
operations forward that beta unchanged.

For a known set of IDs, Anthropic accepts the provider-specific `ids=[...]`
option. The server validates its limits and combinations with other parameters.
Missing or inaccessible IDs are omitted.

## Download generated outputs

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
`headers` as soon as the context is entered. Anthropic response header lookup is
case-insensitive. Both objects remain directly iterable over bytes, so existing
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
paths that cannot be opened raise `InvalidRequestError`. Provider-specific
numeric limits and option combinations are validated by the server, so SDK
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
