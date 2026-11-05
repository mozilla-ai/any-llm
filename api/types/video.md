---
title: Video Types
description: Job, output and parameter models for the video API
---

The job returned by `any_llm.video_generation()`, `any_llm.retrieve_video()` and `any_llm.poll_video()`, and the parameters a provider receives.

**Import:** `from any_llm.types.video import VideoJob, VideoOutput, VideoError`

## Job

`VideoJob.status` is one of `queued`, `in_progress`, `completed` or `failed`; `done` is `True` for the last two.

### `VideoJob`

A video generation job.

Generation is slow, so every provider returns a job first. Poll it with
``retrieve_video`` until ``done``; ``outputs`` is empty until then.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Provider-assigned job ID; pass it to `retrieve_video`. |
| `model` | `str` | Model that renders the video. |
| `status` | `Literal['queued', 'in_progress', 'completed', 'failed']` | Where the job is. Providers' own words map onto these four. |
| `created_at` | `int \| None` | Unix timestamp of the request, when the provider reports it. |
| `completed_at` | `int \| None` | Unix timestamp of completion, when the provider reports it. |
| `progress` | `int \| None` | Percent complete, when the provider reports it. |
| `seconds` | `int \| None` | Length of the video in seconds, when the provider reports it. |
| `size` | `str \| None` | ``WIDTHxHEIGHT`` of the output, when the provider reports it. |
| `error` | `VideoError \| None` | Why a `failed` job failed. |
| `outputs` | `list[VideoOutput]` | The rendered videos; empty until `completed`. |

### `VideoOutput`

One generated video, as a URL, inline base64 data, or both.

| Field | Type | Description |
|-------|------|-------------|
| `url` | `str \| None` | Where the video can be fetched. Some providers' URLs need their credentials; `download_video` adds them. |
| `b64_data` | `str \| None` | The video, base64-encoded, when the provider returns it inline. |
| `mime_type` | `str` | Media type of the video. |

### `VideoError`

Why a video job failed.

| Field | Type | Description |
|-------|------|-------------|
| `message` | `str` | What went wrong, in the provider's words. |
| `code` | `str \| None` | The provider's error code, when it gives one. |

## Parameters

### `VideoGenerationParams`

Parameters for video generation requests.

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | `str` | The provider's own model ID, without a ``provider:`` prefix. |
| `prompt` | `str` | What the video shows. |
| `seconds` | `int \| None` | Length of the video in seconds. |
| `size` | `str \| None` | ``WIDTHxHEIGHT``, such as ``1280x720``. Says both how sharp and what shape; not combined with `resolution` or `aspect_ratio`. |
| `resolution` | `str \| None` | How sharp, by short side: ``480p``, ``720p``, ``1080p`` or ``4k``. Pair it with `aspect_ratio` for the shape. |
| `aspect_ratio` | `str \| None` | What shape, as ``WIDTH:HEIGHT`` such as ``16:9`` or ``9:16``. |
| `seed` | `int \| None` | Seed for reproducible generation. |
| `negative_prompt` | `str \| None` | What the video should not contain. |
| `reference_images` | `list[str] \| None` | Image URLs, provider file URIs or ``data:`` URLs the video takes its subjects or style from. |
| `reference_audios` | `list[str] \| None` | Audio URLs the video follows, for lip sync, beat-matched motion or a sound track. |
| `first_frame` | `str \| None` | Image the video starts from, in the same forms as `reference_images`. |
| `last_frame` | `str \| None` | Image the video ends on, in the same forms as `reference_images`. |
