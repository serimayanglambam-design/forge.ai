import os
MODEL = os.getenv("FORGE_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("FORGE_EMBED", "nomic-embed-text")
TEMPERATURE = 0.2