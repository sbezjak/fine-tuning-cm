"""HF transformers + PEFT backend: the CUDA-path twin of MLXProvider.

The cloud LoRA train (Colab T4, `notebooks/cloud_lora_peft.ipynb`) emits a PEFT
adapter, NOT an MLX adapter - a different on-disk format MLX cannot load. So the
before/after does not fake a conversion; it swaps the BACKEND behind the same
`Provider` seam. This provider loads the base with `adapter_path=None` (the
"before") or the base + PEFT adapter (the "after"), and exposes the identical
`decision_score` read - one forward pass, softmax the last-position logits, sum
each label's token-id family - so the SAME frozen holdout scores through the SAME
`logprob`/`calibration` code, MLX vs PEFT like-for-like.

Two honest port caveats live here, both noted in notes.md:
- bitsandbytes 4-bit is CUDA-only, so LOCAL eval loads the base in fp16/fp32, not
  4-bit. The adapter trained under QLoRA still applies (standard practice); the
  numbers will not match MLX to the decimal, and that gap is itself a finding.
- MLX quantized the base with affine-4bit, the cloud with nf4 - a second reason
  the two backends' scores can diverge slightly.

Imports of torch/transformers/peft are deferred into `_ensure_loaded`, so the
module imports (and the mocked suite runs) without the `peft` group installed.
Deterministic: no sampling, a single forward pass per row.
"""

from __future__ import annotations

from pathlib import Path

from ft_cm.providers.base import Provider


class PEFTProvider(Provider):
    def __init__(
        self,
        model: str,
        adapter_path: str | Path | None = None,
        system: str | None = None,
        max_tokens: int = 8,
    ) -> None:
        self.model_name = model
        self.adapter_path = str(adapter_path) if adapter_path is not None else None
        self.system = system
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None
        self._device = None
        self._label_ids: dict[str, list[int]] | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # MPS (Apple GPU) in fp16 keeps the 1.5B base to ~3 GB - safe on 8 GB and
        # closest to the T4's fp16 compute; fall back to CPU/fp32 where MPS is absent.
        if torch.backends.mps.is_available():
            self._device, dtype = "mps", torch.float16
        else:
            self._device, dtype = "cpu", torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=dtype, low_cpu_mem_usage=True
        )
        if self.adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        self._model = model.to(self._device).eval()

    def unload(self) -> None:
        """Drop the model + free the MPS cache so the next provider holds memory alone."""
        if self._model is None:
            return
        import gc

        import torch

        self._model = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _messages(self, prompt: str) -> list[dict]:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _encode(self, prompt: str) -> dict:
        # return_dict=True -> a BatchEncoding (input_ids + attention_mask), the shape the
        # model wants as **kwargs. (transformers v5 returns a BatchEncoding, not a bare
        # tensor, so we pass it through rather than indexing a plain list.)
        enc = self._tokenizer.apply_chat_template(
            self._messages(prompt),
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return {k: v.to(self._device) for k, v in enc.items()}

    async def generate(self, prompt: str) -> str:
        self._ensure_loaded()
        import torch

        enc = self._encode(prompt)
        with torch.no_grad():
            out = self._model.generate(**enc, max_new_tokens=self.max_tokens, do_sample=False)
        return self._tokenizer.decode(
            out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def decision_score(self, prompt: str):
        """Decision-token confidence, identical read to MLXProvider.decision_score.

        One forward pass; the logits at the LAST position are the model's distribution
        over the answer token. Softmax (in fp32 for a stable sum), add up each label's
        token-id family, hand the per-label mass to the pure restricted-binary readout
        in `logprob`. No generation - reads exactly the position greedy would sample.
        """
        self._ensure_loaded()
        import torch

        from ft_cm.logprob import build_label_token_ids, scores_from_label_probs

        if self._label_ids is None:
            self._label_ids = build_label_token_ids(self._tokenizer)

        enc = self._encode(prompt)
        with torch.no_grad():
            logits = self._model(**enc).logits
        probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
        label_probs = {
            label: float(sum(probs[i].item() for i in id_list))
            for label, id_list in self._label_ids.items()
        }
        return scores_from_label_probs(label_probs)
