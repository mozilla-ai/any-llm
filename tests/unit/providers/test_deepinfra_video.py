import re
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from any_llm import AnyLLM
from any_llm.exceptions import InvalidRequestError, ProviderError, UnsupportedParameterError
from any_llm.providers.deepinfra.deepinfra import DeepinfraProvider
from any_llm.providers.deepinfra.video import (
    VideoSchema,
    _absolute_url,
    convert_inference_response,
    convert_video_params,
)
from any_llm.types.video import VideoGenerationParams, VideoJob

SEEDANCE = "ByteDance/Seedance-2.0"
WAN3 = "Wan-AI/Wan3.0-Video"
WAN26 = "Wan-AI/Wan2.6-T2V"
API_BASE = "https://api.deepinfra.com/v1/openai/"
VIDEO_URL = "https://cdn.deepinfra.com/output/clip.mp4"

# Trimmed copies of the ``in_schema`` DeepInfra publishes at GET /models/{model}.
SEEDANCE_SCHEMA: dict[str, Any] = {
    "properties": {
        "prompt": {"type": "string"},
        "first_frame_image": {"type": "string", "format": "binary"},
        "last_frame_image": {"type": "string", "format": "binary"},
        "reference_images": {"type": "array", "items": {"type": "string"}},
        "reference_audios": {"type": "array", "items": {"type": "string"}},
        "reference_videos": {"type": "array", "items": {"type": "string"}},
        "resolution": {"type": "string", "enum": ["480p", "720p", "1080p"]},
        "aspect_ratio": {"type": "string", "enum": ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"]},
        "duration": {"type": "integer"},
        "seed": {"type": "integer"},
        "generate_audio": {"type": "boolean"},
    }
}
WAN3_SCHEMA: dict[str, Any] = {
    "properties": {
        "prompt": {"type": "string"},
        "media": {"type": "array", "items": {"$ref": "#/definitions/Wan30Media"}},
        "resolution": {"type": "string", "enum": ["480P", "720P", "1080P"]},
        "ratio": {"type": "string", "enum": ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16"]},
        "duration": {"type": "integer"},
        "audio": {"type": "boolean"},
        "seed": {"type": "integer"},
    },
    "definitions": {
        "Wan30Media": {
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["first_frame", "last_frame", "reference_image", "reference_video", "reference_audio"],
                },
                "url": {"type": "string"},
            }
        }
    },
}
WAN26_SCHEMA: dict[str, Any] = {
    "properties": {
        "prompt": {"type": "string"},
        "negative_prompt": {"type": "string"},
        "audio_url": {"type": "string"},
        "size": {"type": "string", "enum": ["1280*720", "720*1280", "960*960"]},
        "duration": {"type": "integer"},
        "seed": {"type": "integer"},
    }
}
PIXVERSE6_SCHEMA: dict[str, Any] = {
    "properties": {
        "prompt": {"type": "string"},
        "generate_audio_switch": {"type": "boolean"},
    }
}
# p-video's ``audio`` is a clip to condition on, not a switch.
P_VIDEO_SCHEMA: dict[str, Any] = {
    "properties": {
        "prompt": {"type": "string"},
        "audio": {"type": "string"},
    }
}
BARE_SCHEMA: dict[str, Any] = {"properties": {"prompt": {"type": "string"}}}


def _schema(model: str = SEEDANCE, in_schema: dict[str, Any] = SEEDANCE_SCHEMA) -> VideoSchema:
    return VideoSchema(model, in_schema)


def _params(model: str = SEEDANCE, **overrides: Any) -> VideoGenerationParams:
    fields: dict[str, Any] = {"model_id": model, "prompt": "a rubber duck"}
    fields.update(overrides)
    return VideoGenerationParams(**fields)


def _convert(params: VideoGenerationParams, schema: VideoSchema, **kwargs: Any) -> dict[str, Any]:
    return convert_video_params(params, schema, "deepinfra", **kwargs)


def _response(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "video_url": VIDEO_URL,
        "status": "ok",
        "out_tokens": 1000,
        "request_id": "R7X9fdlIaF5GlVisBAi5xR3E",
        "inference_status": {"status": "succeeded", "runtime_ms": 120000, "cost": 0.5},
    }
    fields.update(overrides)
    return fields


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> DeepinfraProvider:
    # Both stores are class-wide so module-level poll_video can see them; keep tests apart.
    monkeypatch.setattr(DeepinfraProvider, "_video_jobs", {})
    monkeypatch.setattr(DeepinfraProvider, "_video_schemas", {})
    return DeepinfraProvider(api_key="test-key")


def test_video_schema_reads_a_model_record() -> None:
    schema = VideoSchema.from_model_record(WAN3, {"in_schema": WAN3_SCHEMA})
    assert schema.model == WAN3
    assert schema.enum("resolution") == ["480P", "720P", "1080P"]
    assert schema.enum("duration") is None
    assert schema.enum("missing") is None
    assert schema.media_types() == [
        "first_frame",
        "last_frame",
        "reference_image",
        "reference_video",
        "reference_audio",
    ]


def test_video_schema_has_no_media_types_without_a_media_field() -> None:
    assert _schema().media_types() == []


@pytest.mark.parametrize("record", [{}, {"in_schema": None}, {"in_schema": {"title": "NoProps"}}])
def test_video_schema_rejects_a_record_without_an_input_schema(record: dict[str, Any]) -> None:
    with pytest.raises(ProviderError, match=r"published no input schema for ByteDance/Seedance-2\.0"):
        VideoSchema.from_model_record(SEEDANCE, record)


def test_convert_video_params_sends_only_the_prompt_by_default() -> None:
    assert _convert(_params(), _schema()) == {"prompt": "a rubber duck"}


def test_convert_video_params_maps_every_field_to_seedances_names() -> None:
    params = _params(
        seconds=5,
        size="1280x720",
        seed=7,
        reference_images=["https://img/ref.png"],
        reference_audios=["https://a/beat.mp3"],
        reference_videos=["https://v/moves.mp4", "https://v/turn.mp4"],
        first_frame="data:image/png;base64,Zmlyc3Q=",
        last_frame="data:image/png;base64,bGFzdA==",
        generate_audio=False,
    )
    assert _convert(params, _schema()) == {
        "prompt": "a rubber duck",
        "duration": 5,
        "resolution": "720p",
        "aspect_ratio": "16:9",
        "seed": 7,
        "reference_images": ["https://img/ref.png"],
        "reference_audios": ["https://a/beat.mp3"],
        "reference_videos": ["https://v/moves.mp4", "https://v/turn.mp4"],
        "first_frame_image": "data:image/png;base64,Zmlyc3Q=",
        "last_frame_image": "data:image/png;base64,bGFzdA==",
        "generate_audio": False,
    }


def test_convert_video_params_maps_every_field_to_wan3s_media_items() -> None:
    params = _params(
        WAN3,
        seconds=10,
        size="720x1280",
        seed=7,
        reference_images=["https://img/ref.png"],
        reference_audios=["https://a/beat.mp3"],
        reference_videos=["https://v/moves.mp4", "https://v/turn.mp4"],
        first_frame="https://img/first.png",
        last_frame="https://img/last.png",
        generate_audio=True,
    )
    assert _convert(params, _schema(WAN3, WAN3_SCHEMA)) == {
        "prompt": "a rubber duck",
        "duration": 10,
        "resolution": "720P",
        "ratio": "9:16",
        "seed": 7,
        "audio": True,
        "media": [
            {"type": "first_frame", "url": "https://img/first.png"},
            {"type": "last_frame", "url": "https://img/last.png"},
            {"type": "reference_image", "url": "https://img/ref.png"},
            {"type": "reference_audio", "url": "https://a/beat.mp3"},
            {"type": "reference_video", "url": "https://v/moves.mp4"},
            {"type": "reference_video", "url": "https://v/turn.mp4"},
        ],
    }


def test_convert_video_params_maps_one_reference_audio_to_wan26s_single_clip_field() -> None:
    params = _params(WAN26, reference_audios=["https://a/beat.mp3"])
    assert _convert(params, _schema(WAN26, WAN26_SCHEMA)) == {
        "prompt": "a rubber duck",
        "audio_url": "https://a/beat.mp3",
    }


def test_convert_video_params_rejects_several_reference_audios_for_a_single_clip_field() -> None:
    params = _params(WAN26, reference_audios=["https://a/one.mp3", "https://a/two.mp3"])
    with pytest.raises(InvalidRequestError, match="takes one audio clip as audio_url, not 2"):
        _convert(params, _schema(WAN26, WAN26_SCHEMA))


def test_convert_video_params_maps_size_to_wan26s_exact_sizes() -> None:
    params = _params(WAN26, size="1280x720", negative_prompt="blurry")
    assert _convert(params, _schema(WAN26, WAN26_SCHEMA)) == {
        "prompt": "a rubber duck",
        "negative_prompt": "blurry",
        "size": "1280*720",
    }


def test_convert_video_params_rejects_a_size_wan26_does_not_list() -> None:
    with pytest.raises(InvalidRequestError, match=r"renders 1280\*720, 720\*1280, 960\*960, not '854x480'"):
        _convert(_params(WAN26, size="854x480"), _schema(WAN26, WAN26_SCHEMA))


def test_convert_video_params_maps_generate_audio_to_pixverses_switch() -> None:
    params = _params("Pixverse/Pixverse-6-T2V", generate_audio=False)
    assert _convert(params, _schema("Pixverse/Pixverse-6-T2V", PIXVERSE6_SCHEMA)) == {
        "prompt": "a rubber duck",
        "generate_audio_switch": False,
    }


def test_convert_video_params_rejects_a_keyword_argument_that_contradicts_generate_audio() -> None:
    with pytest.raises(InvalidRequestError, match="generate_audio=False conflicts with audio=True"):
        _convert(_params(WAN3, generate_audio=False), _schema(WAN3, WAN3_SCHEMA), audio=True)


def test_convert_video_params_accepts_a_keyword_argument_that_agrees_with_generate_audio() -> None:
    body = _convert(_params(WAN3, generate_audio=False), _schema(WAN3, WAN3_SCHEMA), audio=False)
    assert body == {"prompt": "a rubber duck", "audio": False}


@pytest.mark.parametrize(
    ("model", "in_schema", "shared", "option", "message"),
    [
        (SEEDANCE, SEEDANCE_SCHEMA, {"seconds": 5}, {"duration": 10}, "seconds=5 conflicts with duration=10"),
        (SEEDANCE, SEEDANCE_SCHEMA, {"seed": 1}, {"seed": 2}, "seed=1 conflicts with seed=2"),
        (WAN26, WAN26_SCHEMA, {"negative_prompt": "blur"}, {"negative_prompt": "rain"}, "negative_prompt='blur'"),
        (SEEDANCE, SEEDANCE_SCHEMA, {"resolution": "720p"}, {"resolution": "1080p"}, "resolution='720p'"),
        (WAN3, WAN3_SCHEMA, {"aspect_ratio": "16:9"}, {"ratio": "9:16"}, "aspect_ratio='16:9' conflicts with ratio"),
        (SEEDANCE, SEEDANCE_SCHEMA, {"size": "1280x720"}, {"resolution": "1080p"}, "size='1280x720'"),
        (WAN26, WAN26_SCHEMA, {"size": "1280x720"}, {"size": "720*1280"}, "size='1280x720' conflicts with size"),
        (SEEDANCE, SEEDANCE_SCHEMA, {"first_frame": "a.png"}, {"first_frame_image": "b.png"}, "first_frame='a.png'"),
        (SEEDANCE, SEEDANCE_SCHEMA, {"reference_images": ["a"]}, {"reference_images": ["b"]}, "reference_images"),
        (SEEDANCE, SEEDANCE_SCHEMA, {"reference_audios": ["a"]}, {"reference_audios": ["b"]}, "reference_audios"),
        (WAN26, WAN26_SCHEMA, {"reference_audios": ["a"]}, {"audio_url": "b"}, "conflicts with audio_url='b'"),
        (SEEDANCE, SEEDANCE_SCHEMA, {"reference_videos": ["a"]}, {"reference_videos": ["b"]}, "reference_videos"),
    ],
)
def test_convert_video_params_rejects_a_keyword_argument_that_contradicts_a_shared_field(
    model: str, in_schema: dict[str, Any], shared: dict[str, Any], option: dict[str, Any], message: str
) -> None:
    with pytest.raises(InvalidRequestError, match=re.escape(message)):
        _convert(_params(model, **shared), _schema(model, in_schema), **option)


def test_convert_video_params_accepts_a_keyword_argument_that_agrees_with_a_shared_field() -> None:
    body = _convert(_params(seconds=5, resolution="720p"), _schema(), duration=5, resolution="720p", fps=24)
    assert body == {"prompt": "a rubber duck", "duration": 5, "resolution": "720p", "fps": 24}


def test_convert_video_params_rejects_a_keyword_argument_that_contradicts_the_prompt() -> None:
    with pytest.raises(InvalidRequestError, match="prompt='a rubber duck' conflicts with prompt='a goose'"):
        _convert(_params(), _schema(), prompt="a goose")


def test_convert_video_params_accepts_a_keyword_argument_that_repeats_the_prompt() -> None:
    assert _convert(_params(), _schema(), prompt="a rubber duck") == {"prompt": "a rubber duck"}


@pytest.mark.parametrize(
    ("shared", "media", "message"),
    [
        (
            {"first_frame": "a.png"},
            [{"type": "first_frame", "url": "b.png"}],
            "first_frame='a.png' conflicts with media[first_frame]=['b.png']",
        ),
        (
            {"first_frame": "a.png"},
            [{"type": "first_frame", "url": "b.png"}, {"type": "first_frame", "url": "a.png"}],
            "first_frame='a.png' conflicts with media[first_frame]=['b.png', 'a.png']",
        ),
        (
            {"reference_images": ["r.png"]},
            [{"type": "reference_image", "url": "s.png"}],
            "reference_images=['r.png'] conflicts with media[reference_image]=['s.png']",
        ),
        (
            {"reference_videos": ["v.mp4"]},
            [{"type": "reference_video", "url": "v.mp4"}, {"type": "reference_video", "url": "w.mp4"}],
            "reference_videos=['v.mp4'] conflicts with media[reference_video]=['v.mp4', 'w.mp4']",
        ),
    ],
)
def test_convert_video_params_rejects_media_items_that_contradict_a_shared_field(
    shared: dict[str, Any], media: list[dict[str, str]], message: str
) -> None:
    with pytest.raises(InvalidRequestError, match=re.escape(message)):
        _convert(_params(WAN3, **shared), _schema(WAN3, WAN3_SCHEMA), media=media)


def test_convert_video_params_keeps_one_copy_of_media_items_the_caller_repeats() -> None:
    body = _convert(
        _params(WAN3, last_frame="a.png", reference_images=["r.png"]),
        _schema(WAN3, WAN3_SCHEMA),
        media=[
            {"type": "last_frame", "url": "a.png"},
            {"type": "reference_image", "url": "r.png"},
            {"type": "reference_video", "url": "v.mp4"},
        ],
    )
    assert body["media"] == [
        {"type": "last_frame", "url": "a.png"},
        {"type": "reference_image", "url": "r.png"},
        {"type": "reference_video", "url": "v.mp4"},
    ]


def test_convert_video_params_rejects_generate_audio_where_audio_is_not_a_switch() -> None:
    with pytest.raises(UnsupportedParameterError, match="'generate_audio' is not supported for deepinfra"):
        _convert(_params("PrunaAI/p-video", generate_audio=True), _schema("PrunaAI/p-video", P_VIDEO_SCHEMA))


def test_convert_video_params_passes_keyword_arguments_through_unchanged() -> None:
    body = _convert(_params(), _schema(), reference_audios=["https://a.mp3"], generate_audio=False)
    assert body == {"prompt": "a rubber duck", "reference_audios": ["https://a.mp3"], "generate_audio": False}


@pytest.mark.parametrize(
    "parameter_name",
    [
        "seconds",
        "seed",
        "negative_prompt",
        "size",
        "first_frame",
        "last_frame",
        "reference_images",
        "reference_audios",
        "reference_videos",
        "generate_audio",
    ],
)
def test_convert_video_params_rejects_a_field_the_schema_has_no_name_for(parameter_name: str) -> None:
    value: Any = "1280x720"
    if parameter_name in {"reference_images", "reference_audios", "reference_videos"}:
        value = ["https://ref"]
    if parameter_name in {"seconds", "seed"}:
        value = 5
    if parameter_name == "generate_audio":
        value = True
    with pytest.raises(UnsupportedParameterError, match=f"'{parameter_name}' is not supported for deepinfra"):
        _convert(_params(**{parameter_name: value}), _schema(SEEDANCE, BARE_SCHEMA))


def test_convert_video_params_rejects_a_negative_prompt_seedance_has_no_field_for() -> None:
    with pytest.raises(UnsupportedParameterError, match="pass the model's own field as a keyword argument"):
        _convert(_params(negative_prompt="blurry"), _schema())


@pytest.mark.parametrize(
    ("size", "resolution", "aspect_ratio"),
    [
        ("854x480", "480p", "16:9"),
        ("864x480", "480p", "16:9"),
        ("640x480", "480p", "4:3"),
        ("720x720", "720p", "1:1"),
        ("720x1280", "720p", "9:16"),
        ("1080x1440", "1080p", "3:4"),
        ("2520x1080", "1080p", "21:9"),
    ],
)
def test_convert_video_params_maps_size_to_resolution_and_aspect_ratio(
    size: str, resolution: str, aspect_ratio: str
) -> None:
    body = _convert(_params(size=size), _schema())
    assert (body["resolution"], body["aspect_ratio"]) == (resolution, aspect_ratio)


def test_convert_video_params_rejects_a_short_side_the_model_does_not_render() -> None:
    with pytest.raises(InvalidRequestError, match="resolution of 480p, 720p, 1080p, not '640x360'"):
        _convert(_params(size="640x360"), _schema())


def test_convert_video_params_rejects_an_aspect_ratio_the_model_does_not_list() -> None:
    with pytest.raises(InvalidRequestError, match="aspect ratio of 16:9, 4:3, 1:1, 3:4, 9:16, not '2520x1080'"):
        _convert(_params(WAN3, size="2520x1080"), _schema(WAN3, WAN3_SCHEMA))


@pytest.mark.parametrize(
    ("model", "in_schema", "resolution", "aspect_ratio", "expected"),
    [
        (SEEDANCE, SEEDANCE_SCHEMA, "480p", "16:9", {"resolution": "480p", "aspect_ratio": "16:9"}),
        (SEEDANCE, SEEDANCE_SCHEMA, "1080P", "9:16", {"resolution": "1080p", "aspect_ratio": "9:16"}),
        (WAN3, WAN3_SCHEMA, "480p", "16:9", {"resolution": "480P", "ratio": "16:9"}),
        (WAN3, WAN3_SCHEMA, "720p", None, {"resolution": "720P"}),
        (WAN3, WAN3_SCHEMA, None, "4:3", {"ratio": "4:3"}),
    ],
)
def test_convert_video_params_maps_resolution_and_aspect_ratio_to_the_models_spelling(
    model: str, in_schema: dict[str, Any], resolution: str | None, aspect_ratio: str | None, expected: dict[str, Any]
) -> None:
    params = _params(model, resolution=resolution, aspect_ratio=aspect_ratio)
    assert _convert(params, _schema(model, in_schema)) == {"prompt": "a rubber duck", **expected}


def test_convert_video_params_rejects_a_resolution_the_model_does_not_render() -> None:
    with pytest.raises(InvalidRequestError, match="resolution of 480p, 720p, 1080p, not '4k 16:9'"):
        _convert(_params(resolution="4k", aspect_ratio="16:9"), _schema())


def test_convert_video_params_rejects_a_tier_for_a_model_that_takes_exact_sizes() -> None:
    with pytest.raises(
        InvalidRequestError, match=r"takes an exact size \(1280\*720, 720\*1280, 960\*960\), so pass size"
    ):
        _convert(_params(WAN26, resolution="720p", aspect_ratio="16:9"), _schema(WAN26, WAN26_SCHEMA))


def test_convert_video_params_rejects_an_aspect_ratio_for_a_model_with_no_ratio_field() -> None:
    in_schema = {"properties": {"prompt": {}, "resolution": {"enum": ["720P", "1080P"]}}}
    with pytest.raises(UnsupportedParameterError, match="'aspect_ratio' is not supported for deepinfra"):
        _convert(_params(resolution="720p", aspect_ratio="16:9"), _schema(SEEDANCE, in_schema))


def test_convert_video_params_rejects_a_resolution_for_a_model_with_no_resolution_field() -> None:
    with pytest.raises(UnsupportedParameterError, match="'resolution' is not supported for deepinfra"):
        _convert(_params(resolution="720p"), _schema(SEEDANCE, BARE_SCHEMA))


def test_convert_video_params_rejects_a_size_when_the_model_has_no_ratio_field() -> None:
    in_schema = {"properties": {"prompt": {}, "resolution": {"enum": ["720P", "1080P"]}}}
    with pytest.raises(InvalidRequestError, match="no shape, so pass resolution, not size '1280x720'"):
        _convert(_params(size="1280x720"), _schema(SEEDANCE, in_schema))


def test_convert_video_params_maps_any_ratio_the_model_lists() -> None:
    in_schema = {"properties": {"prompt": {}, "aspect_ratio": {"enum": ["3:2", "2:3", "²:3", "adaptive"]}}}
    assert _convert(_params(aspect_ratio="3:2"), _schema(SEEDANCE, in_schema)) == {
        "prompt": "a rubber duck",
        "aspect_ratio": "3:2",
    }


def test_convert_video_params_rejects_a_malformed_size() -> None:
    with pytest.raises(InvalidRequestError, match="WIDTHxHEIGHT"):
        _convert(_params(size="720p"), _schema())


def test_absolute_url_keeps_an_absolute_url() -> None:
    assert _absolute_url(VIDEO_URL, API_BASE) == VIDEO_URL


def test_absolute_url_resolves_a_path_against_the_api_host() -> None:
    assert _absolute_url("/model/inference/clip.mp4", API_BASE) == "https://api.deepinfra.com/model/inference/clip.mp4"


def test_convert_inference_response_maps_a_finished_render() -> None:
    job = convert_inference_response(_response(), model=SEEDANCE, api_base=API_BASE)

    assert isinstance(job, VideoJob)
    assert job.id == "R7X9fdlIaF5GlVisBAi5xR3E"
    assert job.model == SEEDANCE
    assert job.status == "completed"
    assert job.done
    assert job.error is None
    assert [output.url for output in job.outputs] == [VIDEO_URL]
    assert job.outputs[0].mime_type == "video/mp4"


def test_convert_inference_response_mints_an_id_when_deepinfra_sends_none() -> None:
    first = convert_inference_response(_response(request_id=None), model=SEEDANCE, api_base=API_BASE)
    second = convert_inference_response(_response(request_id=None), model=SEEDANCE, api_base=API_BASE)

    assert first.id.startswith("deepinfra-")
    assert first.id != second.id


def test_convert_inference_response_treats_a_missing_status_as_finished() -> None:
    job = convert_inference_response(_response(inference_status=None), model=SEEDANCE, api_base=API_BASE)
    assert job.status == "completed"
    assert job.outputs[0].url == VIDEO_URL


@pytest.mark.parametrize(("raw_status", "status"), [("queued", "queued"), ("running", "in_progress")])
def test_convert_inference_response_maps_a_pending_status(raw_status: str, status: str) -> None:
    response = _response(video_url=None, inference_status={"status": raw_status})
    job = convert_inference_response(response, model=SEEDANCE, api_base=API_BASE)
    assert job.status == status
    assert job.outputs == []
    assert job.error is None


def test_convert_inference_response_keeps_outputs_empty_until_the_job_is_complete() -> None:
    response = _response(inference_status={"status": "running"})
    job = convert_inference_response(response, model=SEEDANCE, api_base=API_BASE)
    assert job.status == "in_progress"
    assert job.outputs == []


def test_convert_inference_response_maps_a_failed_render_with_its_detail() -> None:
    response = _response(video_url=None, inference_status={"status": "failed"}, detail="content policy")
    job = convert_inference_response(response, model=SEEDANCE, api_base=API_BASE)

    assert job.status == "failed"
    assert job.error is not None
    assert job.error.message == "content policy"
    assert job.outputs == []


def test_convert_inference_response_fails_a_success_without_a_video() -> None:
    job = convert_inference_response(_response(video_url=""), model=SEEDANCE, api_base=API_BASE)

    assert job.status == "failed"
    assert job.error is not None
    assert job.error.message == "DeepInfra returned no video."


def test_convert_inference_response_reports_the_provider_charge() -> None:
    inference_status = {"status": "succeeded", "cost": 0.38991, "tokens_input": 13, "tokens_generated": 50625}
    job = convert_inference_response(_response(inference_status=inference_status), model=SEEDANCE, api_base=API_BASE)

    assert job.cost == 0.38991
    assert job.tokens == 50638


def test_convert_inference_response_counts_only_the_tokens_reported() -> None:
    inference_status = {"status": "succeeded", "tokens_generated": 50625}
    job = convert_inference_response(_response(inference_status=inference_status), model=SEEDANCE, api_base=API_BASE)

    assert job.cost is None
    assert job.tokens == 50625


def test_convert_inference_response_reports_a_whole_dollar_cost_as_a_float() -> None:
    job = convert_inference_response(
        _response(inference_status={"status": "succeeded", "cost": 1}), model=SEEDANCE, api_base=API_BASE
    )

    assert job.cost == 1.0
    assert isinstance(job.cost, float)


def test_convert_inference_response_reports_no_charge_without_an_inference_status() -> None:
    job = convert_inference_response(_response(inference_status=None), model=SEEDANCE, api_base=API_BASE)

    assert job.cost is None
    assert job.tokens is None


@pytest.mark.parametrize("value", [None, "0.5", True])
def test_convert_inference_response_ignores_a_charge_that_is_not_a_number(value: object) -> None:
    inference_status = {"status": "succeeded", "cost": value, "tokens_input": value, "tokens_generated": value}
    job = convert_inference_response(_response(inference_status=inference_status), model=SEEDANCE, api_base=API_BASE)

    assert job.cost is None
    assert job.tokens is None


def test_convert_inference_response_reports_the_charge_of_a_failed_render() -> None:
    inference_status = {"status": "failed", "cost": 0.0, "tokens_input": 0, "tokens_generated": 0}
    job = convert_inference_response(
        _response(video_url=None, inference_status=inference_status), model=SEEDANCE, api_base=API_BASE
    )

    assert job.status == "failed"
    assert job.cost == 0.0
    assert job.tokens == 0


def test_convert_inference_response_rejects_an_unknown_status() -> None:
    with pytest.raises(ProviderError, match="unknown inference status 'exploded'"):
        convert_inference_response(
            _response(inference_status={"status": "exploded"}), model=SEEDANCE, api_base=API_BASE
        )


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), float("-inf")])
def test_convert_inference_response_ignores_a_cost_that_is_not_finite(cost: float) -> None:
    job = convert_inference_response(
        _response(inference_status={"status": "succeeded", "cost": cost}), model=SEEDANCE, api_base=API_BASE
    )

    assert job.cost is None


def test_provider_supports_video_generation(provider: DeepinfraProvider) -> None:
    assert provider.SUPPORTS_VIDEO_GENERATION is True
    assert DeepinfraProvider.get_provider_metadata().video_generation is True


def test_native_urls_sit_beside_the_openai_endpoint(provider: DeepinfraProvider) -> None:
    assert provider._inference_url(SEEDANCE) == f"https://api.deepinfra.com/v1/inference/{SEEDANCE}"
    assert provider._model_url(SEEDANCE) == f"https://api.deepinfra.com/models/{SEEDANCE}"


def test_native_urls_follow_a_custom_api_base() -> None:
    provider = DeepinfraProvider(api_key="test-key", api_base="https://proxy.example/v1/openai")
    assert provider._inference_url(SEEDANCE) == f"https://proxy.example/v1/inference/{SEEDANCE}"
    assert provider._model_url(SEEDANCE) == f"https://proxy.example/models/{SEEDANCE}"


@pytest.mark.asyncio
async def test_video_generation_fetches_the_schema_then_posts_to_the_inference_endpoint(
    provider: DeepinfraProvider,
) -> None:
    get = AsyncMock(return_value={"in_schema": SEEDANCE_SCHEMA})
    post = AsyncMock(return_value=_response())
    with patch.object(provider.client, "get", get), patch.object(provider.client, "post", post):
        job = await provider.avideo_generation(SEEDANCE, "a rubber duck", seconds=5, size="1280x720")

    assert job.status == "completed"
    assert job.outputs[0].url == VIDEO_URL
    assert get.call_args.args == (f"https://api.deepinfra.com/models/{SEEDANCE}",)
    assert get.call_args.kwargs["options"] == {"timeout": 900.0}
    assert post.call_args.args == (f"https://api.deepinfra.com/v1/inference/{SEEDANCE}",)
    assert post.call_args.kwargs["body"] == {
        "prompt": "a rubber duck",
        "duration": 5,
        "resolution": "720p",
        "aspect_ratio": "16:9",
    }
    assert post.call_args.kwargs["options"] == {"max_retries": 0, "timeout": 900.0}


@pytest.mark.asyncio
async def test_video_generation_fetches_each_models_schema_once(provider: DeepinfraProvider) -> None:
    other = DeepinfraProvider(api_key="other-key")
    get = AsyncMock(return_value={"in_schema": WAN3_SCHEMA})
    post = AsyncMock(return_value=_response())
    with (
        patch.object(provider.client, "get", get),
        patch.object(provider.client, "post", post),
        patch.object(other.client, "get", get),
        patch.object(other.client, "post", post),
    ):
        await provider.avideo_generation(WAN3, "a rubber duck")
        await provider.avideo_generation(WAN3, "a rubber duck")
        # The schema store is class-wide, so a fresh instance reuses it too.
        await other.avideo_generation(WAN3, "a rubber duck")

    assert get.await_count == 1
    assert post.await_count == 3


@pytest.mark.asyncio
async def test_video_generation_fetches_the_schema_again_for_another_api_base(provider: DeepinfraProvider) -> None:
    proxied = DeepinfraProvider(api_key="test-key", api_base="https://proxy.example/v1/openai")
    get = AsyncMock(return_value={"in_schema": WAN3_SCHEMA})
    post = AsyncMock(return_value=_response())
    with (
        patch.object(provider.client, "get", get),
        patch.object(provider.client, "post", post),
        patch.object(proxied.client, "get", get),
        patch.object(proxied.client, "post", post),
    ):
        await provider.avideo_generation(WAN3, "a rubber duck")
        await proxied.avideo_generation(WAN3, "a rubber duck")

    assert [call.args for call in get.await_args_list] == [
        (f"https://api.deepinfra.com/models/{WAN3}",),
        (f"https://proxy.example/models/{WAN3}",),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [{"webhook": "https://hooks.example/done"}, {"extra_body": {"webhook": "https://hooks.example/done"}}],
)
async def test_video_generation_rejects_a_webhook_before_posting(
    provider: DeepinfraProvider, kwargs: dict[str, Any]
) -> None:
    post = AsyncMock(return_value=_response())
    with (
        patch.object(provider.client, "get", AsyncMock(return_value={"in_schema": SEEDANCE_SCHEMA})),
        patch.object(provider.client, "post", post),
        pytest.raises(InvalidRequestError, match="cannot take a webhook"),
    ):
        await provider.avideo_generation(SEEDANCE, "a rubber duck", **kwargs)

    assert post.await_count == 0


@pytest.mark.asyncio
async def test_video_generation_rejects_extra_body_that_contradicts_generate_audio_before_posting(
    provider: DeepinfraProvider,
) -> None:
    post = AsyncMock(return_value=_response())
    with (
        patch.object(provider.client, "get", AsyncMock(return_value={"in_schema": SEEDANCE_SCHEMA})),
        patch.object(provider.client, "post", post),
        pytest.raises(InvalidRequestError, match="generate_audio=False conflicts with generate_audio=True"),
    ):
        await provider.avideo_generation(
            SEEDANCE, "a rubber duck", generate_audio=False, extra_body={"generate_audio": True}
        )

    assert post.await_count == 0


@pytest.mark.asyncio
async def test_video_generation_rejects_extra_body_that_contradicts_seconds_before_posting(
    provider: DeepinfraProvider,
) -> None:
    post = AsyncMock(return_value=_response())
    with (
        patch.object(provider.client, "get", AsyncMock(return_value={"in_schema": SEEDANCE_SCHEMA})),
        patch.object(provider.client, "post", post),
        pytest.raises(InvalidRequestError, match="seconds=5 conflicts with duration=10"),
    ):
        await provider.avideo_generation(SEEDANCE, "a rubber duck", seconds=5, extra_body={"duration": 10})

    assert post.await_count == 0


@pytest.mark.asyncio
async def test_video_generation_rejects_extra_body_that_contradicts_the_prompt_before_posting(
    provider: DeepinfraProvider,
) -> None:
    post = AsyncMock(return_value=_response())
    with (
        patch.object(provider.client, "get", AsyncMock(return_value={"in_schema": SEEDANCE_SCHEMA})),
        patch.object(provider.client, "post", post),
        pytest.raises(InvalidRequestError, match="prompt='a rubber duck' conflicts with prompt='a goose'"),
    ):
        await provider.avideo_generation(SEEDANCE, "a rubber duck", extra_body={"prompt": "a goose"})

    assert post.await_count == 0


@pytest.mark.asyncio
async def test_video_generation_merges_extra_body_media_with_the_mapped_items(provider: DeepinfraProvider) -> None:
    post = AsyncMock(return_value=_response())
    with (
        patch.object(provider.client, "get", AsyncMock(return_value={"in_schema": WAN3_SCHEMA})),
        patch.object(provider.client, "post", post),
    ):
        await provider.avideo_generation(
            WAN3,
            "a rubber duck",
            first_frame="https://img/first.png",
            extra_body={"media": [{"type": "reference_video", "url": "https://v.mp4"}]},
        )

    assert post.call_args.kwargs["body"]["media"] == [
        {"type": "first_frame", "url": "https://img/first.png"},
        {"type": "reference_video", "url": "https://v.mp4"},
    ]


@pytest.mark.asyncio
async def test_video_generation_passes_provider_options_and_extra_body_through(
    provider: DeepinfraProvider,
) -> None:
    get = AsyncMock(return_value={"in_schema": SEEDANCE_SCHEMA})
    post = AsyncMock(return_value=_response())
    with patch.object(provider.client, "get", get), patch.object(provider.client, "post", post):
        await provider.avideo_generation(
            SEEDANCE,
            "a rubber duck",
            generate_audio=True,
            timeout=30,
            extra_headers={"X-Trace": "1"},
            extra_body={"reference_audios": ["https://audio/beat.mp3"]},
        )

    assert post.call_args.kwargs["body"] == {
        "prompt": "a rubber duck",
        "generate_audio": True,
        "reference_audios": ["https://audio/beat.mp3"],
    }
    assert post.call_args.kwargs["options"] == {"max_retries": 0, "timeout": 30, "headers": {"X-Trace": "1"}}
    assert get.call_args.kwargs["options"] == {"timeout": 30}


@pytest.mark.asyncio
async def test_retrieve_video_returns_a_job_started_in_this_process(provider: DeepinfraProvider) -> None:
    with (
        patch.object(provider.client, "get", AsyncMock(return_value={"in_schema": SEEDANCE_SCHEMA})),
        patch.object(provider.client, "post", AsyncMock(return_value=_response())),
    ):
        job = await provider.avideo_generation(SEEDANCE, "a rubber duck")

    assert await provider.aretrieve_video(job.id) is job
    assert (await provider.aretrieve_video(job.id)).cost == 0.5
    # The store is class-wide, so the module-level functions reach it through a fresh instance.
    assert await DeepinfraProvider(api_key="other-key").aretrieve_video(job.id) is job


@pytest.mark.asyncio
async def test_retrieve_video_forgets_the_oldest_jobs_past_the_store_limit(
    provider: DeepinfraProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("any_llm.providers.deepinfra.deepinfra._VIDEO_JOBS_KEPT", 2)
    post = AsyncMock(side_effect=[_response(request_id=f"job-{index}") for index in range(3)])
    with (
        patch.object(provider.client, "get", AsyncMock(return_value={"in_schema": SEEDANCE_SCHEMA})),
        patch.object(provider.client, "post", post),
    ):
        jobs = [await provider.avideo_generation(SEEDANCE, "a rubber duck") for _ in range(3)]

    assert await provider.aretrieve_video("job-1") is jobs[1]
    assert await provider.aretrieve_video("job-2") is jobs[2]
    with pytest.raises(InvalidRequestError, match="only the newest 2 jobs started in this process"):
        await provider.aretrieve_video("job-0")


@pytest.mark.asyncio
async def test_retrieve_video_rejects_a_job_it_has_never_seen(provider: DeepinfraProvider) -> None:
    with pytest.raises(InvalidRequestError, match="jobs started in this process can be retrieved"):
        await provider.aretrieve_video("vid-from-elsewhere")


@pytest.mark.asyncio
async def test_poll_video_finishes_at_once_for_a_synchronous_render(provider: DeepinfraProvider) -> None:
    with (
        patch.object(provider.client, "get", AsyncMock(return_value={"in_schema": SEEDANCE_SCHEMA})),
        patch.object(provider.client, "post", AsyncMock(return_value=_response())),
    ):
        job = await provider.avideo_generation(SEEDANCE, "a rubber duck")

    assert await provider.apoll_video(job.id, poll_interval=0, timeout=1) is job


def test_deepinfra_is_a_folder_provider_not_a_registry_row() -> None:
    assert "deepinfra" not in AnyLLM.get_registry_provider_names()
    assert AnyLLM.get_provider_class("deepinfra") is DeepinfraProvider
