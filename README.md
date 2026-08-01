# bmodel

Helper functions for spinning up LangChain chat models across my homelab apps. Ships with sane defaults pointed at my local llama.cpp servers, and lets any app that installs it override those defaults at runtime.

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

Note these only affect the *built-in* defaults (`chat`/`vision`/`reasoning`/`translation` all currently point at the Gemma4 config). `TRANSLATEGEMMA` and `EMBEDDINGGEMMA` are defined but not wired to a capability yet — reach for `configure()` if you want to use them.

## Supported providers

`ModelConfig.provider` accepts `llama.cpp`, `openai`, or `openrouter`. All three are OpenAI-compatible APIs under the hood, so they're all backed by `ChatOpenAI` — just point `base_url` at the right endpoint.

## API reference

| Name | Description |
|---|---|
| `ModelConfig` | Frozen dataclass describing one model: `provider`, `base_url`, `api_key`, `model`, plus `temperature`, `timeout`, `max_tokens`, `streaming` |
| `ChatModelCapability` | `Literal["chat", "vision", "reasoning", "translation"]` |
| `ModelProvider` | `Literal["llama.cpp", "openai", "openrouter"]` |
| `ModelsAvailable` | `dict[ChatModelCapability, ModelConfig]` |
| `configure(**kwargs)` | Override the default `ModelConfig` for one or more capabilities |
| `reset_defaults()` | Clear all overrides, restoring the built-in defaults |
| `get_model_config(capability, *, model=None)` | Resolve the effective `ModelConfig` for a capability |
| `init_model(*, capability="chat", model=None)` | Build a `BaseChatModel` for any capability |
| `init_chat_model(*, model=None)` | Shortcut for `init_model(capability="chat")` |
| `init_vision_model(*, model=None)` | Shortcut for `init_model(capability="vision")` |
| `init_reasoning_model(*, model=None)` | Shortcut for `init_model(capability="reasoning")` |
| `init_translation_model(*, model=None)` | Shortcut for `init_model(capability="translation")` |
