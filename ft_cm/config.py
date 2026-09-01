"""Environment + run configuration for the fine-tuning study.

`load_dotenv` is carried verbatim from the sibling repos: a keys-only `.env`
loader, deliberately tiny instead of the python-dotenv dependency, values already
set in the real environment win. Only needed once a paid judge is wired; the
local MLX + Ollama path uses no keys.

The model + path constants live here so switching the base model (the planned
0.5B -> 1.5B swap) is a one-line change, not a hunt through the scripts.
"""

from __future__ import annotations

import os
from pathlib import Path

# The base model is a config var by design: smoke-test on 0.5B, then re-run the
# identical pipeline on 1.5B to feel the 8 GB ceiling firsthand. 4-bit quantized
# MLX community builds, the Apple-Silicon LoRA path.
BASE_MODEL = os.environ.get("FT_CM_BASE_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
BASE_MODEL_STRETCH = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

ADAPTER_PATH = Path(os.environ.get("FT_CM_ADAPTER_PATH", "adapters/smoke"))
SMOKE_DATA_DIR = Path("data/smoke")
REAL_DATA_DIR = Path("data/real")


def load_dotenv(path: Path | str = ".env") -> None:
    env = Path(path)
    if not env.is_file():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
