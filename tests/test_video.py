import asyncio
import base64
import json
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.messages import AIMessage, HumanMessage, SystemMessage

from bmodel.base import ModelConfig
from bmodel.config import configure_video_model
from bmodel.video import (
    ExtractedFrame,
    LlamaCppVideoAdapter,
    VideoInfo,
    init_video_model,
)

FFPROBE_JSON = json.dumps(
    {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "duration": "10.0",
                "avg_frame_rate": "30/1",
            }
        ],
        "format": {"duration": "10.0"},
    }
)


def _make_model_config(model: str = "testing-vision-model") -> ModelConfig:
    return ModelConfig(
        provider="openai",
        base_url="http://localhost:8080/v1",
        api_key="testing",
        model=model,
        supports_vision=True,
        supports_audio=True,
    )


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr("bmodel.video.shutil.which", lambda name: f"/usr/bin/{name}")

    return LlamaCppVideoAdapter(model=MagicMock(), model_config=_make_model_config())


@pytest.fixture
def audio_adapter(monkeypatch):
    monkeypatch.setattr("bmodel.video.shutil.which", lambda name: f"/usr/bin/{name}")

    return LlamaCppVideoAdapter(
        model=MagicMock(),
        model_config=_make_model_config(),
        include_audio=True,
        audio_model=MagicMock(),
        audio_model_config=_make_model_config("testing-audio-model"),
    )


# --- constructor validation ---


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"frames_per_second": 0}, "frames_per_second"),
        ({"frames_per_second": -1}, "frames_per_second"),
        ({"max_frames": 0}, "max_frames"),
        ({"max_width": 0}, "max_width"),
        ({"image_format": "gif"}, "image_format"),
        ({"image_quality": 1}, "image_quality"),
        ({"image_quality": 32}, "image_quality"),
        ({"audio_format": "flac"}, "audio_format"),
        ({"max_audio_seconds": 0}, "max_audio_seconds"),
        ({"max_audio_seconds": -1}, "max_audio_seconds"),
        ({"command_timeout_seconds": 0}, "command_timeout_seconds"),
    ],
)
def test_constructor_rejects_invalid_arguments(monkeypatch, kwargs, message):
    monkeypatch.setattr("bmodel.video.shutil.which", lambda name: f"/usr/bin/{name}")

    with pytest.raises(ValueError, match=message):
        LlamaCppVideoAdapter(model=MagicMock(), **kwargs)


def test_missing_binary_raises(monkeypatch):
    monkeypatch.setattr("bmodel.video.shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        LlamaCppVideoAdapter(model=MagicMock())


def test_include_audio_defaults_to_none_audio_model(adapter):
    assert adapter.include_audio is False
    assert adapter.audio_model is None


def test_include_audio_resolves_audio_model_via_init_audio_model(monkeypatch):
    monkeypatch.setattr("bmodel.video.shutil.which", lambda name: f"/usr/bin/{name}")

    with patch("bmodel.video.init_audio_model", return_value=MagicMock()) as mock_init_audio:
        adapter = LlamaCppVideoAdapter(
            model=MagicMock(),
            include_audio=True,
            audio_model_config=_make_model_config("configured-audio-model"),
        )

    mock_init_audio.assert_called_once()
    assert adapter.audio_model is not None


# --- _effective_fps ---


def test_effective_fps_uses_configured_rate_for_short_video(adapter):
    assert adapter._effective_fps(10.0) == pytest.approx(1.0)


def test_effective_fps_reduces_rate_for_long_video(adapter):
    # max_frames=60 default; 600s at 1fps needs 600 frames, so the rate
    # must drop to 60/600 = 0.1 to keep frames spread across the video.
    assert adapter._effective_fps(600.0) == pytest.approx(0.1)


def test_effective_fps_falls_back_when_duration_unknown(adapter):
    assert adapter._effective_fps(0.0) == pytest.approx(adapter.frames_per_second)


# --- static parsing helpers ---


def test_parse_frame_rate_valid():
    assert LlamaCppVideoAdapter._parse_frame_rate("30/1") == pytest.approx(30.0)


def test_parse_frame_rate_invalid_inputs():
    assert LlamaCppVideoAdapter._parse_frame_rate(None) is None
    assert LlamaCppVideoAdapter._parse_frame_rate("") is None
    assert LlamaCppVideoAdapter._parse_frame_rate("not-a-fraction") is None
    assert LlamaCppVideoAdapter._parse_frame_rate("30/0") is None


def test_optional_int_valid_and_invalid():
    assert LlamaCppVideoAdapter._optional_int("1920") == 1920
    assert LlamaCppVideoAdapter._optional_int(None) is None
    assert LlamaCppVideoAdapter._optional_int("not-a-number") is None


# --- _run ---


def test_run_returns_result_on_success(adapter):
    fake_result = subprocess.CompletedProcess(args=["x"], returncode=0, stdout="ok", stderr="")

    with patch("bmodel.video.subprocess.run", return_value=fake_result):
        result = adapter._run(["ffmpeg", "-version"])

    assert result.stdout == "ok"


def test_run_raises_on_nonzero_exit(adapter):
    fake_result = subprocess.CompletedProcess(args=["x"], returncode=1, stdout="", stderr="boom")

    with patch("bmodel.video.subprocess.run", return_value=fake_result):
        with pytest.raises(RuntimeError, match="exit code 1"):
            adapter._run(["ffmpeg", "-version"])


def test_run_raises_on_timeout(adapter):
    with patch(
        "bmodel.video.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1),
    ):
        with pytest.raises(RuntimeError, match="timed out"):
            adapter._run(["ffmpeg", "-version"])


# --- probe ---


def test_probe_parses_video_stream(adapter, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")
    fake_result = MagicMock(stdout=FFPROBE_JSON)

    with patch.object(adapter, "_run", return_value=fake_result) as mock_run:
        info = adapter.probe(video_path)

    mock_run.assert_called_once()
    assert info.duration_seconds == pytest.approx(10.0)
    assert info.width == 1920
    assert info.height == 1080
    assert info.fps == pytest.approx(30.0)


def test_probe_raises_for_missing_file(adapter, tmp_path):
    with pytest.raises(FileNotFoundError):
        adapter.probe(tmp_path / "missing.mp4")


def test_probe_raises_when_no_video_stream(adapter, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")
    fake_result = MagicMock(stdout=json.dumps({"streams": [], "format": {}}))

    with patch.object(adapter, "_run", return_value=fake_result):
        with pytest.raises(ValueError, match="No video stream"):
            adapter.probe(video_path)


# --- extract_frames ---


def test_extract_frames_creates_extracted_frame_list(adapter, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")
    output_dir = tmp_path / "frames"
    fake_info = VideoInfo(path=video_path, duration_seconds=3.0, width=100, height=100, fps=30.0)

    def fake_run(command):
        for index in range(1, 4):
            (output_dir / f"frame-{index:06d}.jpg").write_bytes(b"fake-image-bytes")
        return MagicMock()

    with (
        patch.object(adapter, "probe", return_value=fake_info),
        patch.object(adapter, "_run", side_effect=fake_run),
    ):
        frames = adapter.extract_frames(video_path, output_dir)

    assert len(frames) == 3
    assert all(isinstance(frame, ExtractedFrame) for frame in frames)
    assert frames[0].timestamp_seconds == pytest.approx(0.0)


def test_extract_frames_raises_when_no_frames_written(adapter, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")
    output_dir = tmp_path / "frames"
    fake_info = VideoInfo(path=video_path, duration_seconds=3.0, width=100, height=100, fps=30.0)

    with (
        patch.object(adapter, "probe", return_value=fake_info),
        patch.object(adapter, "_run", return_value=MagicMock()),
    ):
        with pytest.raises(RuntimeError, match="did not extract"):
            adapter.extract_frames(video_path, output_dir)


def test_extract_frames_supports_png_format(monkeypatch, tmp_path):
    monkeypatch.setattr("bmodel.video.shutil.which", lambda name: f"/usr/bin/{name}")
    adapter = LlamaCppVideoAdapter(model=MagicMock(), image_format="png")

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")
    output_dir = tmp_path / "frames"
    fake_info = VideoInfo(path=video_path, duration_seconds=3.0, width=100, height=100, fps=30.0)
    commands = []

    def fake_run(command):
        commands.append(command)
        (output_dir / "frame-000001.png").write_bytes(b"fake-image-bytes")
        return MagicMock()

    with (
        patch.object(adapter, "probe", return_value=fake_info),
        patch.object(adapter, "_run", side_effect=fake_run),
    ):
        frames = adapter.extract_frames(video_path, output_dir)

    assert len(frames) == 1
    assert frames[0].path == output_dir / "frame-000001.png"
    assert "-q:v" not in commands[0]


# --- extract_audio ---


def test_extract_audio_creates_audio_file(adapter, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")
    output_dir = tmp_path / "audio"
    fake_info = VideoInfo(path=video_path, duration_seconds=3.0, width=100, height=100, fps=30.0)

    def fake_run(command):
        (output_dir / "audio.mp3").write_bytes(b"fake-audio-bytes")
        return MagicMock()

    with (
        patch.object(adapter, "probe", return_value=fake_info),
        patch.object(adapter, "_run", side_effect=fake_run),
    ):
        audio_path = adapter.extract_audio(video_path, output_dir)

    assert audio_path == output_dir / "audio.mp3"
    assert audio_path.read_bytes() == b"fake-audio-bytes"


def test_extract_audio_raises_when_no_audio_written(adapter, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")
    output_dir = tmp_path / "audio"
    fake_info = VideoInfo(path=video_path, duration_seconds=3.0, width=100, height=100, fps=30.0)

    with (
        patch.object(adapter, "probe", return_value=fake_info),
        patch.object(adapter, "_run", return_value=MagicMock()),
    ):
        with pytest.raises(RuntimeError, match="did not extract any audio"):
            adapter.extract_audio(video_path, output_dir)


def test_extract_audio_supports_wav_format(monkeypatch, tmp_path):
    monkeypatch.setattr("bmodel.video.shutil.which", lambda name: f"/usr/bin/{name}")
    adapter = LlamaCppVideoAdapter(model=MagicMock(), audio_format="wav")

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")
    output_dir = tmp_path / "audio"
    fake_info = VideoInfo(path=video_path, duration_seconds=3.0, width=100, height=100, fps=30.0)

    def fake_run(command):
        (output_dir / "audio.wav").write_bytes(b"fake-audio-bytes")
        return MagicMock()

    with (
        patch.object(adapter, "probe", return_value=fake_info),
        patch.object(adapter, "_run", side_effect=fake_run),
    ):
        audio_path = adapter.extract_audio(video_path, output_dir)

    assert audio_path == output_dir / "audio.wav"


# --- _audio_to_block ---


def test_audio_to_block_encodes_mp3(tmp_path):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"fake-audio-bytes")

    block = LlamaCppVideoAdapter._audio_to_block(audio_path, "mp3")

    assert block["type"] == "audio"
    # Must be exactly "audio/mp3", not mimetypes' "audio/mpeg" — LangChain's
    # OpenAI translator derives the API `format` field via mime_type.split("/"),
    # and OpenAI only accepts literally "mp3" or "wav" there.
    assert block["mime_type"] == "audio/mp3"
    assert block["base64"] == base64.b64encode(b"fake-audio-bytes").decode("ascii")


def test_audio_to_block_encodes_wav(tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake-audio-bytes")

    block = LlamaCppVideoAdapter._audio_to_block(audio_path, "wav")

    assert block["mime_type"] == "audio/wav"


# --- _frame_to_block ---


def test_frame_to_block_encodes_image(tmp_path):
    frame_path = tmp_path / "frame-000001.jpg"
    frame_path.write_bytes(b"fake-image-bytes")
    frame = ExtractedFrame(path=frame_path, timestamp_seconds=1.0)

    block = LlamaCppVideoAdapter._frame_to_block(frame)

    assert block["type"] == "image"
    assert block["mime_type"] == "image/jpeg"
    assert block["base64"] == base64.b64encode(b"fake-image-bytes").decode("ascii")


def test_frame_to_block_raises_for_unknown_mime_type(tmp_path):
    frame_path = tmp_path / "frame-000001.unknownext"
    frame_path.write_bytes(b"fake-image-bytes")
    frame = ExtractedFrame(path=frame_path, timestamp_seconds=1.0)

    with pytest.raises(ValueError, match="Could not determine MIME type"):
        LlamaCppVideoAdapter._frame_to_block(frame)


# --- build_messages ---


def test_build_messages_includes_text_and_image_blocks(adapter, tmp_path):
    frame_path = tmp_path / "frame-000001.jpg"
    frame_path.write_bytes(b"fake-image-bytes")
    frames = [ExtractedFrame(path=frame_path, timestamp_seconds=0.5)]

    messages = adapter.build_messages(prompt="Describe it", frames=frames)

    assert len(messages) == 2
    system_message, human_message = messages
    assert isinstance(system_message, SystemMessage)
    assert isinstance(human_message, HumanMessage)
    assert len(human_message.content) == 3
    assert human_message.content[0]["type"] == "text"
    assert human_message.content[1]["type"] == "text"
    assert human_message.content[2]["type"] == "image"


def test_build_messages_includes_audio_block_when_provided(adapter, tmp_path):
    frame_path = tmp_path / "frame-000001.jpg"
    frame_path.write_bytes(b"fake-image-bytes")
    frames = [ExtractedFrame(path=frame_path, timestamp_seconds=0.5)]
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"fake-audio-bytes")
    audio_block = LlamaCppVideoAdapter._audio_to_block(audio_path, "mp3")

    messages = adapter.build_messages(prompt="Describe it", frames=frames, audio_block=audio_block)

    _, human_message = messages
    assert len(human_message.content) == 5
    assert human_message.content[3]["type"] == "text"
    assert human_message.content[4]["type"] == "audio"


# --- invoke / ainvoke ---


def test_invoke_orchestrates_extract_and_model_call(adapter, tmp_path):
    frame_path = tmp_path / "frame-000001.jpg"
    frame_path.write_bytes(b"fake-image-bytes")
    frames = [ExtractedFrame(path=frame_path, timestamp_seconds=0.0)]
    adapter.model.invoke.return_value = AIMessage(content="a description")

    with patch.object(adapter, "extract_frames", return_value=frames) as mock_extract:
        result = adapter.invoke("video.mp4")

    mock_extract.assert_called_once()
    adapter.model.invoke.assert_called_once()
    assert result.content == "a description"


def test_ainvoke_orchestrates_extract_and_model_call(adapter, tmp_path):
    frame_path = tmp_path / "frame-000001.jpg"
    frame_path.write_bytes(b"fake-image-bytes")
    frames = [ExtractedFrame(path=frame_path, timestamp_seconds=0.0)]
    adapter.model.ainvoke = AsyncMock(return_value=AIMessage(content="a description"))

    with patch.object(adapter, "extract_frames", return_value=frames):
        result = asyncio.run(adapter.ainvoke("video.mp4"))

    adapter.model.ainvoke.assert_called_once()
    assert result.content == "a description"


def test_invoke_routes_to_audio_model_when_include_audio(audio_adapter, tmp_path):
    frame_path = tmp_path / "frame-000001.jpg"
    frame_path.write_bytes(b"fake-image-bytes")
    frames = [ExtractedFrame(path=frame_path, timestamp_seconds=0.0)]
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"fake-audio-bytes")
    audio_adapter.audio_model.invoke.return_value = AIMessage(content="a description")

    with (
        patch.object(audio_adapter, "extract_frames", return_value=frames),
        patch.object(audio_adapter, "extract_audio", return_value=audio_path) as mock_extract_audio,
    ):
        result = audio_adapter.invoke("video.mp4")

    mock_extract_audio.assert_called_once()
    audio_adapter.audio_model.invoke.assert_called_once()
    audio_adapter.model.invoke.assert_not_called()
    assert result.content == "a description"


def test_ainvoke_routes_to_audio_model_when_include_audio(audio_adapter, tmp_path):
    frame_path = tmp_path / "frame-000001.jpg"
    frame_path.write_bytes(b"fake-image-bytes")
    frames = [ExtractedFrame(path=frame_path, timestamp_seconds=0.0)]
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"fake-audio-bytes")
    audio_adapter.audio_model.ainvoke = AsyncMock(return_value=AIMessage(content="a description"))

    with (
        patch.object(audio_adapter, "extract_frames", return_value=frames),
        patch.object(audio_adapter, "extract_audio", return_value=audio_path),
    ):
        result = asyncio.run(audio_adapter.ainvoke("video.mp4"))

    audio_adapter.audio_model.ainvoke.assert_called_once()
    assert result.content == "a description"


# --- init_video_model ---


def test_init_video_model_constructs_adapter(monkeypatch):
    monkeypatch.setattr("bmodel.video.shutil.which", lambda name: f"/usr/bin/{name}")

    with patch("bmodel.video.build_chat_model", return_value=MagicMock()):
        adapter = init_video_model(frames_per_second=2.0, max_frames=10, max_width=512)

    assert isinstance(adapter, LlamaCppVideoAdapter)
    assert adapter.frames_per_second == 2.0
    assert adapter.max_frames == 10
    assert adapter.max_width == 512


def test_init_video_model_respects_configure_video_model(monkeypatch):
    monkeypatch.setattr("bmodel.video.shutil.which", lambda name: f"/usr/bin/{name}")
    override = _make_model_config("configured-video-model")
    configure_video_model(override)

    with patch("bmodel.video.build_chat_model", return_value=MagicMock()) as mock_build:
        init_video_model()

    mock_build.assert_called_once_with(override)


def test_init_video_model_rejects_video_model_without_vision_support(monkeypatch):
    monkeypatch.setattr("bmodel.video.shutil.which", lambda name: f"/usr/bin/{name}")
    text_only = ModelConfig(
        provider="openai",
        base_url="http://localhost:8080/v1",
        api_key="testing",
        model="text-only-model",
    )

    with pytest.raises(ValueError, match="does not declare vision support"):
        init_video_model(model=text_only)


def test_init_video_model_passes_through_audio_options(monkeypatch):
    monkeypatch.setattr("bmodel.video.shutil.which", lambda name: f"/usr/bin/{name}")
    audio_override = _make_model_config("configured-audio-model")

    with (
        patch("bmodel.video.build_chat_model", return_value=MagicMock()),
        patch("bmodel.video.init_audio_model", return_value=MagicMock()) as mock_init_audio,
    ):
        adapter = init_video_model(
            include_audio=True,
            audio_model=audio_override,
            audio_format="wav",
            max_audio_seconds=120.0,
        )

    mock_init_audio.assert_called_once()
    _, kwargs = mock_init_audio.call_args
    assert kwargs["model"] == audio_override
    assert adapter.include_audio is True
    assert adapter.audio_format == "wav"
    assert adapter.max_audio_seconds == 120.0


def test_init_audio_model_rejects_audio_model_without_audio_support(monkeypatch):
    monkeypatch.setattr("bmodel.video.shutil.which", lambda name: f"/usr/bin/{name}")
    text_only = ModelConfig(
        provider="openai",
        base_url="http://localhost:8080/v1",
        api_key="testing",
        model="text-only-model",
    )

    with pytest.raises(ValueError, match="does not declare audio support"):
        init_video_model(include_audio=True, audio_model=text_only)
