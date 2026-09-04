"""The backend seam. Carried from the sibling repos, narrowed to what a classifier
eval needs: a single async `generate(prompt) -> str`. Scorers and the eval harness
treat every backend identically through this method, so the same held-out slice
runs against the base model, the LoRA-tuned model, or a live Ollama check without
the harness knowing which is behind it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Provider(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str: ...

    def unload(self) -> None:
        """Release the loaded model so a second provider can hold memory alone.
        Default no-op; backends that pin a large model override it. Used by the
        before/after eval to keep only ONE model resident at a time (base scored
        and freed before the tuned model loads), which matters on an 8 GB Mac
        where two 1.5B models would thrash."""
        return
