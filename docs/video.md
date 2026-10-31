---
title: Video Generation
description: Generate videos from a prompt and images as a job you poll
---

## Overview

The video API turns a text prompt, optionally with a first frame, a last frame or reference images, into a short video. Rendering takes from tens of seconds to several minutes, so every provider returns a **job** first:

1. `video_generation()` starts the job and returns a `VideoJob`, usually `queued` or `in_progress`.
2. `retrieve_video()` reads the job again. `poll_video()` does that for you until the job is `completed` or `failed`.
3. `download_video()` returns the finished video as bytes, whether the provider handed back a URL or inline data.

Each function has an async twin with an `a` prefix: `avideo_generation()`, `aretrieve_video()`, `apoll_video()` and `adownload_video()`.

## Supported Providers

| Provider | `SUPPORTS_VIDEO_GENERATION` | Notes |
|----------|-----------------------------|-------|
| Gemini | Yes | Gemini Omni Flash (`gemini-omni-1.1-flash`) through the Interactions API, created with `background=True` and read back with `interactions.get`. `size` maps to Omni's `16:9`/`9:16` aspect ratio and `360p`/`720p`/`1080p`/`4k` resolution, so the short side must be 360, 720, 1080 or 2160 pixels and the video cannot be square. `first_frame` and `last_frame` go in as image parts before the prompt, in that order, which is how Omni assigns their roles; `reference_images` follow them. `seed` passes through. `seconds` and `negative_prompt` raise `UnsupportedParameterError`; put what to avoid in the prompt. The finished video arrives inline (base64) or as a Files API URI, which `download_video` fetches with your API key. |
| Together | Yes | `POST /v2/videos` and `GET /v2/videos/{id}` through the Together SDK, for models such as `ByteDance/Seedance-2.0`. `size` maps to `width`/`height`, `seconds` to the string Together expects, and `first_frame`/`last_frame` to `frame_images`. Together-only options such as `fps`, `steps`, `guidance_scale` and `output_format` pass through as keyword arguments. Seedance renders fixed resolutions only (for example `864x496`, `1280x720`, `1920x1080`, and their portrait twins); a `size` that is not `WIDTHxHEIGHT` raises `InvalidRequestError` before any request is sent, and any other `WIDTHxHEIGHT` is accepted, but the job then fails with code `unsupportedModelResolution` and the full list in `error.message`. The finished video is a public URL. |

All other providers return `SUPPORTS_VIDEO_GENERATION = False` and raise `NotImplementedError` if called. OpenAI is among them: OpenAI removed the Sora 2 models and the Videos API on September 24, 2026, with no replacement.

## Installation

Each video provider needs its own extra:

```bash
pip install any-llm-sdk[gemini]
pip install any-llm-sdk[together]
```

## Quick Start

```python
from any_llm import download_video, poll_video, video_generation

job = video_generation(
    "together:ByteDance/Seedance-2.0",
    prompt="A yellow rubber duck sits on a wooden table while sunlight slowly moves across it.",
    seconds=5,
    size="1280x720",
)
print(job.id, job.status)

job = poll_video("together", job.id, poll_interval=10, timeout=900)
if job.status == "failed":
    raise RuntimeError(job.error.message if job.error else "video generation failed")

video = download_video("together", job.outputs[0])
print(f"{len(video)} bytes of {job.outputs[0].mime_type}")
```

Write `video` to a file with the extension its `mime_type` implies, `.mp4` for `video/mp4`.

The same call works against Gemini by changing the model string and dropping `seconds`, which Omni Flash does not take:

```python
job = video_generation("gemini:gemini-omni-1.1-flash", prompt="...", size="1280x720")
```

### Frames and references

```python
job = video_generation(
    "gemini:gemini-omni-1.1-flash",
    prompt="A smooth cinematic transition from sunrise to a starry night.",
    first_frame="https://example.com/sunrise.png",
    last_frame="data:image/png;base64,....",
    reference_images=["https://example.com/style.png"],
)
```

Images are URLs, provider file URIs, or `data:` URLs with base64 content. A provider that cannot take one of them says so with `UnsupportedParameterError`.

### Async

```python
import asyncio
from any_llm import adownload_video, apoll_video, avideo_generation

async def main() -> bytes:
    job = await avideo_generation("together:ByteDance/Seedance-2.0", prompt="penguins building a snowman")
    job = await apoll_video("together", job.id)
    if job.status == "failed":
        raise RuntimeError(job.error.message if job.error else "video generation failed")
    return await adownload_video("together", job.outputs[0])

asyncio.run(main())
```

## API Reference

### `video_generation()` / `avideo_generation()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Provider-prefixed model ID (e.g. `"together:ByteDance/Seedance-2.0"`) |
| `prompt` | `str` | required | What the video shows |
| `seconds` | `int \| None` | `None` | Length of the video in seconds |
| `size` | `str \| None` | `None` | `WIDTHxHEIGHT`, such as `"1280x720"` |
| `seed` | `int \| None` | `None` | Seed for reproducible generation |
| `negative_prompt` | `str \| None` | `None` | What the video should not contain |
| `reference_images` | `list[str] \| None` | `None` | Images the video takes its subjects or style from |
| `first_frame` | `str \| None` | `None` | Image the video starts from |
| `last_frame` | `str \| None` | `None` | Image the video ends on |
| `provider` | `str \| None` | `None` | Explicit provider name. Inferred from `model` if omitted. |
| `api_key` | `str \| None` | `None` | Provider API key. Falls back to the environment variable. |
| `api_base` | `str \| None` | `None` | Provider API base URL. Falls back to the environment variable. |
| `client_args` | `dict \| None` | `None` | Additional arguments for the provider client constructor |

Any other keyword argument is passed to the provider's API call unchanged.

### `retrieve_video()` / `aretrieve_video()`

Take `provider` and `video_id` and return the job as it is now.

### `poll_video()` / `apoll_video()`

Take `provider` and `video_id`, retrieve the job every `poll_interval` seconds (default 5) and return it once it is `completed` or `failed`. A `failed` job is returned, not raised, so its `error` can be read. After `timeout` seconds (default 600; `None` waits forever) they raise `TimeoutError`.

### `download_video()` / `adownload_video()`

Take `provider` and one entry of `VideoJob.outputs` and return the video as bytes. Inline data is decoded; a URL is fetched, with the provider's credentials where its URLs need them.

### `VideoJob`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Provider-assigned job ID; pass it to `retrieve_video` |
| `model` | `str` | Model that renders the video |
| `status` | `"queued" \| "in_progress" \| "completed" \| "failed"` | Where the job is. Providers' own words map onto these four. |
| `done` | `bool` | `True` once `status` is `completed` or `failed` |
| `created_at` | `int \| None` | Unix timestamp of the request |
| `completed_at` | `int \| None` | Unix timestamp of completion, when the provider reports it |
| `progress` | `int \| None` | Percent complete, when the provider reports it |
| `seconds` | `int \| None` | Length of the video, when the provider reports it |
| `size` | `str \| None` | `WIDTHxHEIGHT` of the video, when the provider reports it |
| `error` | `VideoError \| None` | Why a `failed` job failed |
| `outputs` | `list[VideoOutput]` | The rendered videos; empty until `completed` |

### `VideoOutput`

| Field | Type | Description |
|-------|------|-------------|
| `url` | `str \| None` | Where the video can be fetched |
| `b64_data` | `str \| None` | The video, base64-encoded, when the provider returns it inline |
| `mime_type` | `str` | Media type of the video, `video/mp4` unless the provider says otherwise |
