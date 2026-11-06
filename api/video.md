---
title: Video
description: Start, poll and download video generation jobs
---

Video generation is slow, so `video_generation` returns a `VideoJob` rather than a video. `retrieve_video` reads the job again, `poll_video` waits for it, and `download_video` fetches the result. See [Video Generation](../video.md) for the providers and what each one takes.

## Start a job

### `any_llm.video_generation()`

```
def video_generation(
    model: str,
    prompt: str,
    *,
    provider: str | LLMProvider | None = None,
    seconds: int | None = None,
    size: str | None = None,
    resolution: str | None = None,
    aspect_ratio: str | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    reference_images: list[str] | None = None,
    reference_audios: list[str] | None = None,
    reference_videos: list[str] | None = None,
    first_frame: str | None = None,
    last_frame: str | None = None,
    generate_audio: bool | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> VideoJob
```

### `any_llm.avideo_generation()`

Async variant with the same parameters.

```
async def avideo_generation(
    model: str,
    prompt: str,
    *,
    provider: str | LLMProvider | None = None,
    seconds: int | None = None,
    size: str | None = None,
    resolution: str | None = None,
    aspect_ratio: str | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    reference_images: list[str] | None = None,
    reference_audios: list[str] | None = None,
    reference_videos: list[str] | None = None,
    first_frame: str | None = None,
    last_frame: str | None = None,
    generate_audio: bool | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> VideoJob
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | *required* | Model identifier. **Recommended**: Use with separate `provider` parameter (e.g., model='ByteDance/Seedance-2.0', provider='together'). **Alternative**: Combined format 'provider:model' (e.g., 'together:ByteDance/Seedance-2.0'). |
| `prompt` | `str` | *required* | A text description of the desired video. |
| `provider` | `str \| LLMProvider \| None` | None | Provider name to use for the request (e.g., 'together'). |
| `seconds` | `int \| None` | None | Length of the video in seconds. |
| `size` | `str \| None` | None | Size of the video as ``WIDTHxHEIGHT`` (e.g., '1280x720'). Not combined with the next two. |
| `resolution` | `str \| None` | None | How sharp, by short side: '480p', '720p', '1080p' or '4k'. |
| `aspect_ratio` | `str \| None` | None | What shape, as ``WIDTH:HEIGHT`` (e.g., '16:9'). |
| `seed` | `int \| None` | None | Seed for reproducible generation. |
| `negative_prompt` | `str \| None` | None | What the video should not contain. |
| `reference_images` | `list[str] \| None` | None | Image URLs the video should take its subjects or style from. |
| `reference_audios` | `list[str] \| None` | None | Audio URLs the video should follow, for lip sync, beat-matched motion or a sound track. |
| `reference_videos` | `list[str] \| None` | None | Video URLs the video should take its motion, subjects or style from. |
| `first_frame` | `str \| None` | None | Image URL to start the video from. |
| `last_frame` | `str \| None` | None | Image URL to end the video on. |
| `generate_audio` | `bool \| None` | None | Whether the video carries a sound track. None leaves the provider's default. |
| `api_key` | `str \| None` | None | API key for the provider. |
| `api_base` | `str \| None` | None | Base URL for the provider API. |
| `client_args` | `dict[str, Any] \| None` | None | Additional provider-specific arguments for client instantiation. |
| `**kwargs` | `Any` | *required* | Additional provider-specific arguments. |

## Read a job

### `any_llm.retrieve_video()`

```
def retrieve_video(
    provider: str | LLMProvider,
    video_id: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> VideoJob
```

### `any_llm.aretrieve_video()`

Async variant with the same parameters.

```
async def aretrieve_video(
    provider: str | LLMProvider,
    video_id: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> VideoJob
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str \| LLMProvider` | *required* | Provider name to use for the request (e.g., 'together') |
| `video_id` | `str` | *required* | The ID of the video job to retrieve |
| `api_key` | `str \| None` | None | API key for the provider |
| `api_base` | `str \| None` | None | Base URL for the provider API |
| `client_args` | `dict[str, Any] \| None` | None | Additional provider-specific arguments for client instantiation |
| `**kwargs` | `Any` | *required* | Additional provider-specific arguments |

## Wait for a job

### `any_llm.poll_video()`

```
def poll_video(
    provider: str | LLMProvider,
    video_id: str,
    *,
    poll_interval: float = 5.0,
    timeout: float | None = 600.0,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> VideoJob
```

### `any_llm.apoll_video()`

Async variant with the same parameters.

```
async def apoll_video(
    provider: str | LLMProvider,
    video_id: str,
    *,
    poll_interval: float = 5.0,
    timeout: float | None = 600.0,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> VideoJob
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str \| LLMProvider` | *required* | Provider name to use for the request (e.g., 'together') |
| `video_id` | `str` | *required* | The ID of the video job to wait for |
| `poll_interval` | `float` | 5.0 | Seconds to wait between retrievals |
| `timeout` | `float \| None` | 600.0 | Seconds to wait in total before giving up, or `None` to wait forever |
| `api_key` | `str \| None` | None | API key for the provider |
| `api_base` | `str \| None` | None | Base URL for the provider API |
| `client_args` | `dict[str, Any] \| None` | None | Additional provider-specific arguments for client instantiation |
| `**kwargs` | `Any` | *required* | Additional provider-specific arguments passed to each retrieval |

## Download a video

### `any_llm.download_video()`

```
def download_video(
    provider: str | LLMProvider,
    output: VideoOutput,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> bytes
```

### `any_llm.adownload_video()`

Async variant with the same parameters.

```
async def adownload_video(
    provider: str | LLMProvider,
    output: VideoOutput,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> bytes
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str \| LLMProvider` | *required* | Provider that generated the video (e.g., 'together') |
| `output` | `VideoOutput` | *required* | One entry of `VideoJob.outputs` |
| `api_key` | `str \| None` | None | API key for the provider |
| `api_base` | `str \| None` | None | Base URL for the provider API |
| `client_args` | `dict[str, Any] \| None` | None | Additional provider-specific arguments for client instantiation |
| `**kwargs` | `Any` | *required* | Additional provider-specific arguments |

## Usage

```python
from any_llm import download_video, poll_video, video_generation

job = video_generation("together:ByteDance/Seedance-2.0", prompt="penguins building a snowman", seconds=5)
job = poll_video("together", job.id, poll_interval=10, timeout=900)
if job.status == "completed":
    video = download_video("together", job.outputs[0])
    print(f"{len(video)} bytes of {job.outputs[0].mime_type}")
```

{% hint style="info" %}
Not all providers support video generation. Check the [providers page](../providers.md) for support details, or query `ProviderMetadata.video_generation` programmatically. A provider that does not support it raises `NotImplementedError`.
{% endhint %}
