from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.content import (
    AudioContentBlock,
    ImageContentBlock,
    TextContentBlock,
    create_audio_block,
    create_image_block,
    create_text_block,
)

from .audio import init_audio_model
from .base import ModelConfig
from .chat import build_chat_model
from .config import get_video_model_config

DEFAULT_SYSTEM_PROMPT = (
    "You analyze sampled video frames. Infer temporal changes carefully. "
    "Do not claim to see motion or hear audio that isn't supported by the "
    "supplied frames, and clearly distinguish observations from inference."
)

DEFAULT_PROMPT = "Describe what happens throughout this video in chronological order."


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration_seconds: float
    width: int | None
    height: int | None
    fps: float | None


@dataclass(frozen=True)
class ExtractedFrame:
    path: Path
    timestamp_seconds: float


class LlamaCppVideoAdapter:
    """
    Analyze video with a vision-capable chat model.

    The chat model never receives the raw video. This adapter:
      1. probes the video with ffprobe,
      2. extracts sampled frames with ffmpeg,
      3. sends those frames to the configured vision model as image content
         blocks.
    """

    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        model_config: ModelConfig | None = None,
        frames_per_second: float = 1.0,
        max_frames: int = 60,
        max_width: int = 1024,
        image_format: str = "jpeg",
        image_quality: int = 3,
        include_audio: bool = False,
        audio_model: BaseChatModel | None = None,
        audio_model_config: ModelConfig | None = None,
        audio_format: str = "mp3",
        max_audio_seconds: float = 600.0,
        command_timeout_seconds: float = 300.0,
    ) -> None:
        if frames_per_second <= 0:
            raise ValueError("frames_per_second must be greater than zero")

        if max_frames <= 0:
            raise ValueError("max_frames must be greater than zero")

        if max_width <= 0:
            raise ValueError("max_width must be greater than zero")

        if image_format not in {"jpeg", "png"}:
            raise ValueError("image_format must be 'jpeg' or 'png'")

        if not (2 <= image_quality <= 31):
            raise ValueError("image_quality must be between 2 and 31")

        if audio_format not in {"mp3", "wav"}:
            raise ValueError("audio_format must be 'mp3' or 'wav'")

        if max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be greater than zero")

        if command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be greater than zero")

        self.model = model or build_chat_model(
            get_video_model_config(model=model_config),
        )
        self.frames_per_second = frames_per_second
        self.max_frames = max_frames
        self.max_width = max_width
        self.image_format = image_format
        self.image_quality = image_quality
        self.command_timeout_seconds = command_timeout_seconds

        self.include_audio = include_audio
        self.audio_format = audio_format
        self.max_audio_seconds = max_audio_seconds
        self.audio_model = None

        if include_audio:
            self.audio_model = audio_model or init_audio_model(
                model=audio_model_config,
            )

        self._require_binary("ffmpeg")
        self._require_binary("ffprobe")

    @staticmethod
    def _require_binary(name: str) -> None:
        if shutil.which(name) is None:
            raise RuntimeError(
                f"`{name}` was not found on PATH. Install FFmpeg first.")

    def _run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.command_timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"Command timed out after {self.command_timeout_seconds}s:\n"
                f"{' '.join(command)}"
            ) from error

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"Command failed with exit code {result.returncode}:\n"
                f"{' '.join(command)}\n\n{stderr}"
            )

        return result

    def probe(self, video_path: str | Path) -> VideoInfo:
        path = Path(video_path).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")

        result = self._run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ]
        )

        data = json.loads(result.stdout)

        video_stream = next(
            (
                stream
                for stream in data.get("streams", [])
                if stream.get("codec_type") == "video"
            ),
            None,
        )

        if video_stream is None:
            raise ValueError(f"No video stream found in: {path}")

        duration_raw = (
            video_stream.get("duration") or data.get(
                "format", {}).get("duration") or 0
        )

        fps = self._parse_frame_rate(
            video_stream.get("avg_frame_rate") or video_stream.get(
                "r_frame_rate")
        )

        return VideoInfo(
            path=path,
            duration_seconds=float(duration_raw),
            width=self._optional_int(video_stream.get("width")),
            height=self._optional_int(video_stream.get("height")),
            fps=fps,
        )

    @staticmethod
    def _optional_int(value: object) -> int | None:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_frame_rate(value: object) -> float | None:
        if not isinstance(value, str) or not value:
            return None

        try:
            numerator, denominator = value.split("/", maxsplit=1)
            denominator_value = float(denominator)

            if denominator_value == 0:
                return None

            return float(numerator) / denominator_value
        except (ValueError, ZeroDivisionError):
            return None

    def _effective_fps(self, duration_seconds: float) -> float:
        # Without this, a video longer than max_frames / frames_per_second
        # would only ever get frames extracted from its start, since
        # `-frames:v` stops ffmpeg the instant it hits max_frames. Slowing
        # the sampling rate down keeps frames spread across the whole video.
        if duration_seconds <= 0:
            return self.frames_per_second

        max_supported_fps = self.max_frames / duration_seconds

        return min(self.frames_per_second, max_supported_fps)

    def extract_frames(
        self,
        video_path: str | Path,
        output_directory: str | Path,
    ) -> list[ExtractedFrame]:
        info = self.probe(video_path)
        effective_fps = self._effective_fps(info.duration_seconds)

        output_dir = Path(output_directory).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        extension = "jpg" if self.image_format == "jpeg" else "png"
        pattern = output_dir / f"frame-%06d.{extension}"

        # Avoid upscale: shrink only when the source is wider than max_width.
        scale_filter = (
            f"scale='min({self.max_width},iw)':-2:force_original_aspect_ratio=decrease"
        )

        video_filter = f"fps={effective_fps},{scale_filter}"

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(info.path),
            "-vf",
            video_filter,
            "-frames:v",
            str(self.max_frames),
        ]

        if self.image_format == "jpeg":
            command.extend(["-q:v", str(self.image_quality)])

        command.append(str(pattern))

        self._run(command)

        frame_paths = sorted(output_dir.glob(f"frame-*.{extension}"))

        if not frame_paths:
            raise RuntimeError("FFmpeg did not extract any frames")

        return [
            ExtractedFrame(
                path=frame_path,
                timestamp_seconds=index / effective_fps,
            )
            for index, frame_path in enumerate(frame_paths)
        ]

    def extract_audio(
        self,
        video_path: str | Path,
        output_directory: str | Path,
    ) -> Path:
        info = self.probe(video_path)

        output_dir = Path(output_directory).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        extension = "mp3" if self.audio_format == "mp3" else "wav"
        output_path = output_dir / f"audio.{extension}"

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(info.path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-t",
            str(self.max_audio_seconds),
        ]

        if self.audio_format == "mp3":
            command.extend(["-codec:a", "libmp3lame", "-q:a", "4"])
        else:
            command.extend(["-codec:a", "pcm_s16le"])

        command.append(str(output_path))

        self._run(command)

        if not output_path.is_file():
            raise RuntimeError("FFmpeg did not extract any audio")

        return output_path

    @staticmethod
    def _audio_to_block(audio_path: Path, audio_format: str) -> AudioContentBlock:
        # Deliberately not using mimetypes.guess_type() here: it returns the
        # IANA-registered MIME type ("audio/mpeg" for .mp3, "audio/x-wav" for
        # .wav), but LangChain's OpenAI translator derives the API's
        # `format` field by naively splitting mime_type on "/" — "mpeg"/
        # "x-wav" aren't valid values (OpenAI only accepts "mp3" or "wav"),
        # so the real MIME type breaks the request. audio_format is already
        # constrained to exactly "mp3"/"wav", so build the tag from that.
        encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")

        return create_audio_block(
            base64=encoded,
            mime_type=f"audio/{audio_format}",
        )

    @staticmethod
    def _frame_to_block(frame: ExtractedFrame) -> ImageContentBlock:
        mime_type, _ = mimetypes.guess_type(frame.path.name)

        if mime_type is None:
            raise ValueError(f"Could not determine MIME type for {frame.path}")

        encoded = base64.b64encode(frame.path.read_bytes()).decode("ascii")

        return create_image_block(
            base64=encoded,
            mime_type=mime_type,
        )

    def build_messages(
        self,
        *,
        prompt: str,
        frames: Sequence[ExtractedFrame],
        audio_block: AudioContentBlock | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> list[SystemMessage | HumanMessage]:
        content: list[TextContentBlock | ImageContentBlock | AudioContentBlock] = [
            create_text_block(
                f"{prompt}\n\n"
                f"The video is represented by {len(frames)} sampled frames. "
                "Each image is preceded by its approximate timestamp."
            )
        ]

        for frame in frames:
            content.append(
                create_text_block(
                    f"Frame at approximately {frame.timestamp_seconds:.2f} seconds:"
                )
            )
            content.append(self._frame_to_block(frame))

        if audio_block is not None:
            content.append(create_text_block("Full audio track from the video:"))
            content.append(audio_block)

        return [
            SystemMessage(system_prompt),
            # LangChain's typed content-block helpers (TypedDicts) aren't
            # structurally assignable to HumanMessage's looser `str | dict`
            # content type per PEP 589 — they're equivalent dicts at runtime.
            HumanMessage(content=cast("list[str | dict]", content)),
        ]

    async def ainvoke(
        self,
        video_path: str | Path,
        *,
        prompt: str = DEFAULT_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> AIMessage:
        with tempfile.TemporaryDirectory(prefix="llama-video-") as temp_dir:
            frames = await asyncio.to_thread(
                self.extract_frames,
                video_path,
                temp_dir,
            )

            audio_block = None
            model = self.model

            if self.include_audio:
                assert self.audio_model is not None

                audio_path = await asyncio.to_thread(
                    self.extract_audio,
                    video_path,
                    temp_dir,
                )
                audio_block = self._audio_to_block(audio_path, self.audio_format)
                model = self.audio_model

            messages = self.build_messages(
                prompt=prompt,
                frames=frames,
                audio_block=audio_block,
                system_prompt=system_prompt,
            )

            return await model.ainvoke(messages)

    def invoke(
        self,
        video_path: str | Path,
        *,
        prompt: str = DEFAULT_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> AIMessage:
        with tempfile.TemporaryDirectory(prefix="llama-video-") as temp_dir:
            frames = self.extract_frames(video_path, temp_dir)

            audio_block = None
            model = self.model

            if self.include_audio:
                assert self.audio_model is not None

                audio_path = self.extract_audio(video_path, temp_dir)
                audio_block = self._audio_to_block(audio_path, self.audio_format)
                model = self.audio_model

            messages = self.build_messages(
                prompt=prompt,
                frames=frames,
                audio_block=audio_block,
                system_prompt=system_prompt,
            )

            return model.invoke(messages)


def init_video_model(
    *,
    model: ModelConfig | None = None,
    frames_per_second: float = 1.0,
    max_frames: int = 60,
    max_width: int = 1024,
    include_audio: bool = False,
    audio_model: ModelConfig | None = None,
    audio_format: str = "mp3",
    max_audio_seconds: float = 600.0,
) -> LlamaCppVideoAdapter:
    return LlamaCppVideoAdapter(
        model_config=model,
        frames_per_second=frames_per_second,
        max_frames=max_frames,
        max_width=max_width,
        include_audio=include_audio,
        audio_model_config=audio_model,
        audio_format=audio_format,
        max_audio_seconds=max_audio_seconds,
    )
