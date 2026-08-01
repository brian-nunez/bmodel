import os

MODEL_GEMMA4_URL = os.getenv(
    "MODEL_GEMMA4_URL",
    "http://10.0.0.119:8080/v1",
)
MODEL_GEMMA4_API_KEY = os.getenv(
    "MODEL_GEMMA4_API_KEY",
    "testing",
)
MODEL_GEMMA4_MODEL_ID = os.getenv(
    "MODEL_GEMMA4_MODEL_ID",
    "ggml-org/gemma-4-E2B-it-GGUF:Q8_0",
)

MODEL_TRANSLATEGEMMA_URL = os.getenv(
    "MODEL_TRANSLATEGEMMA_URL",
    "http://10.0.0.119:8080/v1",
)
MODEL_TRANSLATEGEMMA_API_KEY = os.getenv(
    "MODEL_TRANSLATEGEMMA_API_KEY",
    "testing",
)
MODEL_TRANSLATEGEMMA_MODEL_ID = os.getenv(
    "MODEL_TRANSLATEGEMMA_MODEL_ID",
    "ggml-org/gemma-4-E2B-it-GGUF:Q8_0",
)

MODEL_EMBEDDINGGEMMA_URL = os.getenv(
    "MODEL_EMBEDDINGGEMMA_URL",
    "http://10.0.0.119:8082/v1",
)
MODEL_EMBEDDINGGEMMA_API_KEY = os.getenv(
    "MODEL_EMBEDDINGGEMMA_API_KEY",
    "testing",
)
MODEL_EMBEDDINGGEMMA_MODEL_ID = os.getenv(
    "MODEL_EMBEDDINGGEMMA_MODEL_ID",
    "unsloth/embeddinggemma-300m-GGUF:Q8_0",
)
