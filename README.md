# bmodel

Helper functions for spinning up LangChain chat, embedding, video, and audio models across my homelab apps. Ships with sane defaults pointed at my local llama.cpp servers, and lets any app that installs it override those defaults at runtime.

## Installation

Hosted in two places — GitHub is primary, with a self-hosted mirror recommended for installs.

**Recommended (self-hosted mirror):**

```bash
uv add git+https://git.b8z.me/brian/bmodel
```

**Primary (GitHub):**

```bash
uv add git+https://github.com/brian-nunez/bmodel
```

For local development against a checked-out copy of this repo:

```bash
uv add --editable /path/to/bmodel
```

Persisting conversation state to SQLite or Postgres pulls in extra drivers, so those are optional extras — only install them if you need that backend:

```bash
uv add "bmodel[sqlite] @ git+https://git.b8z.me/brian/bmodel"
uv add "bmodel[postgres] @ git+https://git.b8z.me/brian/bmodel"
```

## Quick start

```python
from bmodel import init_chat_model

model = init_chat_model()
response = model.invoke("Hello!")
```

`init_chat_model()` returns a LangChain `BaseChatModel` — use it like any other LangChain chat model (`.invoke()`, `.stream()`, `.batch()`, etc.).

Other capability-specific helpers work the same way:

```python
from bmodel import init_vision_model, init_reasoning_model, init_translation_model

vision_model = init_vision_model()
reasoning_model = init_reasoning_model()
translation_model = init_translation_model()
```

Or use `init_model()` directly if you want to pass the capability as a value instead of picking a named function:

```python
from bmodel import init_model

model = init_model(capability="chat")
```

## Configuring defaults

Out of the box, every capability (`chat`, `vision`, `reasoning`, `translation`) points at the same local Gemma model. Call `configure()` once at your app's startup to point any capability at a different model — every `init_*_model()` call after that picks up the override automatically:

```python
from bmodel import configure, ModelConfig

configure(
    chat_model=ModelConfig(
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_key="sk-...",
        model="gpt-4o",
    ),
)
```

`configure()` only touches the capabilities you pass in — anything you leave out keeps its current value. Call it again later to change more.

To go back to the built-in defaults (useful in tests, so one test's `configure()` call doesn't leak into the next):

```python
from bmodel import reset_defaults

reset_defaults()
```

You can also override a model for a single call without touching the shared defaults at all, by passing `model=` directly:

```python
from bmodel import init_chat_model, ModelConfig

model = init_chat_model(
    model=ModelConfig(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-...",
        model="anthropic/claude-sonnet-4.5",
    ),
)
```

Precedence, highest to lowest: explicit `model=` argument → `configure()` override → built-in default.

### Vision and audio require a capability flag

Any model used for vision, video, or audio input must have the matching flag set on its `ModelConfig` — `supports_vision=True` and/or `supports_audio=True`. Both default to `False`. This is a self-declared check (it verifies you *said* the model supports it, not that it's actually true), but it stops a text-only model from being silently used somewhere it can't work:

```python
ModelConfig(
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4o",
    supports_vision=True,
    supports_audio=True,
)
```

## Embeddings

```python
from bmodel import init_embedding_model

embeddings = init_embedding_model()
vector = embeddings.embed_query("hello world")
vectors = embeddings.embed_documents(["doc one", "doc two"])
```

`init_embedding_model()` returns a LangChain `Embeddings` instance, defaulting to a local EmbeddingGemma model. Override it the same way as chat models, with `configure_embedding_model()` for a shared default or `model=` for a single call:

```python
from bmodel import configure_embedding_model, ModelConfig

configure_embedding_model(
    ModelConfig(provider="openai", base_url="...", api_key="sk-...", model="text-embedding-3-small"),
)
```

Compare embedding vectors with `similarity()`:

```python
from bmodel import similarity

score = similarity(vector_a, vector_b)                    # cosine (default)
score = similarity(vector_a, vector_b, metric="dot")       # dot product
score = similarity(vector_a, vector_b, metric="euclidean") # inverted distance, still "higher = more similar"
```

Pure Python, no dependencies — meant for learning/prototyping and one-off scripts, not production-scale vector search. For real search over many documents, push the comparison into the database instead (`pgvector`, `sqlite-vec`) rather than comparing in a Python loop.

## Video

Analyzes a local video file: samples still frames from it with `ffmpeg`/`ffprobe` (must be on `PATH`) and sends them to a chat model as image content — the model never receives the raw video file itself. Requires a `ModelConfig` with `supports_vision=True`.

```python
import asyncio
from bmodel import init_video_model

async def main():
    video_model = init_video_model()
    result = await video_model.ainvoke(
        "/path/to/video.mp4",
        prompt="What happens in this video?",
    )
    print(result.content)

asyncio.run(main())
```

`invoke()` (sync) works the same way. Override the default with `configure_video_model()` or pass `model=` for a single call.

Tunable frame sampling:

```python
video_model = init_video_model(
    frames_per_second=2.0,  # sampling rate — auto-reduced for long videos so frames still span the whole thing
    max_frames=30,          # hard cap on total frames extracted
    max_width=768,          # frames are downscaled to this width, never upscaled
)
```

Pass `include_audio=True` to also extract the video's audio track and send it alongside the frames in the same request, if the model accepts audio input too (requires `supports_audio=True`). By default this reuses the same model as the frames; pass `audio_model=` to use a different `ModelConfig` for the audio specifically. The audio track is capped at `max_audio_seconds` (default 600s) before extraction:

```python
video_model = init_video_model(include_audio=True, max_audio_seconds=120.0)
```

## Audio

```python
from bmodel import init_audio_model

audio_model = init_audio_model()
```

Returns a plain LangChain `BaseChatModel`, defaulting to a local Gemma model. Requires a `ModelConfig` with `supports_audio=True`. To send actual audio, build a `HumanMessage` with an audio content block yourself:

```python
import base64
from langchain.messages import HumanMessage
from langchain_core.messages.content import create_audio_block

audio_bytes = open("clip.mp3", "rb").read()
message = HumanMessage(content=[
    {"type": "text", "text": "What is said in this clip?"},
    create_audio_block(base64=base64.b64encode(audio_bytes).decode("ascii"), mime_type="audio/mp3"),
])

result = audio_model.invoke([message])
```

Override the default with `configure_audio_model()`, or pass `model=` for a single call.

## Agents

```python
import asyncio
from bmodel import init_agent

async def main():
    agent = await init_agent(tools=[])
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "hi"}]},
        config={"configurable": {"thread_id": "conversation-1"}},
    )
    print(result)

asyncio.run(main())
```

`init_agent()` is a thin wrapper around LangChain's `create_agent()`: it resolves a chat model and a checkpointer, then returns whatever `create_agent()` returns, untouched — every method on the result (`.ainvoke()`, `.invoke()`, `.astream()`, `.get_state()`, etc.) works exactly as documented upstream, nothing bmodel-specific to learn. `tools` is required; any other `create_agent()` keyword argument (`system_prompt`, `response_format`, `middleware`, and so on) is passed straight through.

`init_agent()` is itself `async`, so it must be awaited — `agent = await init_agent(...)`, not `agent = init_agent(...)`. This is because building a persistent checkpointer (below) can require an awaited database connection.

By default the agent uses the `chat` capability's model, same as `init_chat_model()`. Pass `model=` to use a different `ModelConfig` for just this agent:

```python
from bmodel import init_agent, ModelConfig

agent = await init_agent(
    tools=[],
    model=ModelConfig(
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_key="sk-...",
        model="gpt-4o",
    ),
)
```

### Checkpointing

Agents persist conversation state — keyed by the `thread_id` you pass in `config={"configurable": {"thread_id": ...}}` — via a LangGraph checkpointer. Three backends are supported: `memory` (the default, no setup required), `sqlite`, and `postgres`.

Pass a `CheckpointerConfig` to use `sqlite` or `postgres` for a single agent:

```python
from bmodel import init_agent, CheckpointerConfig

agent = await init_agent(
    tools=[],
    checkpointer=CheckpointerConfig(
        backend="sqlite",
        sqlite_path="/data/checkpoints.db",
    ),
)
```

`postgres` requires `postgres_url`:

```python
agent = await init_agent(
    tools=[],
    checkpointer=CheckpointerConfig(
        backend="postgres",
        postgres_url="postgresql://user:pass@10.0.0.5:5432/bmodel",
    ),
)
```

Set a shared default once at your app's startup with `configure_checkpointer()` — every `init_agent()` call after that picks it up automatically, the same pattern as `configure()` for chat models:

```python
from bmodel import configure_checkpointer, CheckpointerConfig

configure_checkpointer(
    CheckpointerConfig(
        backend="sqlite",
        sqlite_path="/data/checkpoints.db",
    )
)

agent = await init_agent(tools=[])  # uses the configured sqlite backend
```

Or skip configuration entirely and pass an already-built LangGraph checkpointer instance directly:

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = await init_agent(
    tools=[],
    checkpointer=InMemorySaver(),
)
```

`sqlite` and `postgres` checkpointers are async by default (`CheckpointerConfig.async_mode=True`), matching `init_agent()`'s own async construction. Set `async_mode=False` if the agent will only ever be called synchronously (`.invoke()`, never `.ainvoke()`) — a checkpointer built for one mode raises `NotImplementedError` if used in the other. `memory` supports both modes natively, so `async_mode` has no effect on it.

The `sqlite` and `postgres` backends need their own drivers installed — `uv add "bmodel[sqlite] @ ..."` or `uv add "bmodel[postgres] @ ..."` — `memory` needs nothing extra.

## Environment variables

The built-in defaults themselves read from environment variables at import time, so you can point them at a different server without writing any Python:

| Variable | Default |
|---|---|
| `MODEL_GEMMA4_URL` | `http://10.0.0.119:8080/v1` |
| `MODEL_GEMMA4_API_KEY` | `testing` |
| `MODEL_GEMMA4_MODEL_ID` | `ggml-org/gemma-4-E2B-it-GGUF:Q8_0` |
| `MODEL_TRANSLATEGEMMA_URL` | `http://10.0.0.119:8080/v1` |
| `MODEL_TRANSLATEGEMMA_API_KEY` | `testing` |
| `MODEL_TRANSLATEGEMMA_MODEL_ID` | `ggml-org/gemma-4-E2B-it-GGUF:Q8_0` |
| `MODEL_EMBEDDINGGEMMA_URL` | `http://10.0.0.119:8082/v1` |
| `MODEL_EMBEDDINGGEMMA_API_KEY` | `testing` |
| `MODEL_EMBEDDINGGEMMA_MODEL_ID` | `unsloth/embeddinggemma-300m-GGUF:Q8_0` |

Note these only affect the *built-in* defaults. `chat`/`vision`/`reasoning`/`translation` and the video/audio defaults all currently point at the Gemma4 config; the embedding default points at `EMBEDDINGGEMMA_CONFIG`. `TRANSLATEGEMMA_CONFIG` is defined but not wired to any default — reach for `configure()` if you want to use it.

## Supported providers

`ModelConfig.provider` accepts `llama.cpp`, `openai`, or `openrouter`. All three are OpenAI-compatible APIs under the hood, so they're all backed by `ChatOpenAI` (or `OpenAIEmbeddings` for embeddings) — just point `base_url` at the right endpoint.

## API reference

| Name | Description |
|---|---|
| `ModelConfig` | Frozen dataclass describing one model: `provider`, `base_url`, `api_key`, `model`, plus `temperature`, `timeout`, `max_tokens`, `streaming`, `supports_vision`, `supports_audio` |
| `ChatModelCapability` | `Literal["chat", "vision", "reasoning", "translation"]` |
| `ModelProvider` | `Literal["llama.cpp", "openai", "openrouter"]` |
| `SimilarityMetric` | `Literal["cosine", "dot", "euclidean"]` |
| `ModelsAvailable` | `dict[ChatModelCapability, ModelConfig]` |
| `CheckpointerConfig` | Frozen dataclass describing one checkpointer: `backend` (`"memory"` \| `"sqlite"` \| `"postgres"`), `sqlite_path`, `postgres_url`, `async_mode` (default `True`) |
| `CheckpointerBackend` | `Literal["memory", "sqlite", "postgres"]` |
| `configure(**kwargs)` | Override the default `ModelConfig` for one or more capabilities |
| `configure_embedding_model(model)` | Override the default embedding `ModelConfig` |
| `configure_video_model(model)` | Override the default video `ModelConfig` |
| `configure_audio_model(model)` | Override the default audio `ModelConfig` |
| `configure_checkpointer(config)` | Override the default `CheckpointerConfig` used by `init_agent()` |
| `reset_defaults()` | Clear all overrides, restoring the built-in defaults |
| `get_model_config(capability, *, model=None)` | Resolve the effective `ModelConfig` for a capability |
| `get_embedding_model_config(*, model=None)` | Resolve the effective embedding `ModelConfig` |
| `get_video_model_config(*, model=None)` | Resolve the effective video `ModelConfig` |
| `get_audio_model_config(*, model=None)` | Resolve the effective audio `ModelConfig` |
| `get_checkpointer_config(*, config=None)` | Resolve the effective `CheckpointerConfig` |
| `init_model(*, capability="chat", model=None)` | Build a `BaseChatModel` for any capability |
| `init_chat_model(*, model=None)` | Shortcut for `init_model(capability="chat")` |
| `init_vision_model(*, model=None)` | Shortcut for `init_model(capability="vision")` |
| `init_reasoning_model(*, model=None)` | Shortcut for `init_model(capability="reasoning")` |
| `init_translation_model(*, model=None)` | Shortcut for `init_model(capability="translation")` |
| `init_embedding_model(*, model=None)` | Build an `Embeddings` instance |
| `similarity(a, b, *, metric="cosine")` | Compare two embedding vectors |
| `init_audio_model(*, model=None)` | Build a `BaseChatModel` for a `supports_audio=True` model |
| `init_video_model(*, model=None, ...)` | Build a `LlamaCppVideoAdapter` |
| `LlamaCppVideoAdapter` | `.invoke(path, *, prompt=..., system_prompt=...)` / `.ainvoke(...)` — samples frames (and optionally audio) from a video file and sends them to a chat model |
| `VideoInfo` | Dataclass: `path`, `duration_seconds`, `width`, `height`, `fps` — result of `LlamaCppVideoAdapter.probe()` |
| `ExtractedFrame` | Dataclass: `path`, `timestamp_seconds` — one sampled video frame |
| `init_checkpointer(*, config=None)` | `async` — build a LangGraph `BaseCheckpointSaver` from a `CheckpointerConfig` |
| `init_agent(*, tools, model=None, checkpointer=None, **kwargs)` | `async` — build an agent via `create_agent()`; `checkpointer` accepts a `CheckpointerConfig`, a raw `BaseCheckpointSaver`, or `None` for the configured default |
