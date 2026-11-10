"""Qwen2.5-VL backend via Hugging Face transformers (same model family as the Tx).

One instance is loaded per run and shared by Tx, Rx and (optionally) the judge,
so both agents run on exactly the same weights. Requires the ``model`` extra
(see ``requirements-model.txt``) and, in practice, a GPU.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from .base import ChatMessage, Generation, GenerationConfig


class HFTokenizer:
    """Wraps a Hugging Face tokenizer; special tokens are never added."""

    def __init__(self, hf_tokenizer: Any, name: str) -> None:
        self._tok = hf_tokenizer
        self.name = name
        self.vocab_size = len(hf_tokenizer)

    @classmethod
    def from_pretrained(cls, model_id: str) -> "HFTokenizer":
        """Load only the tokenizer (no weights); used by the analysis scripts."""
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(model_id), model_id)

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def token_strings(self, ids: Sequence[int]) -> list[str]:
        return [self._tok.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in ids]


class QwenBackend:
    """Chat generation with Qwen2.5-VL; supports text-only and image turns."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", device_map: str = "auto", torch_dtype: str = "auto") -> None:
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as error:  # pragma: no cover - depends on optional extra
            raise ImportError("Install requirements-model.txt to use the Qwen backend") from error
        self._torch = torch
        self._process_vision_info = process_vision_info
        self.model_id = model_id
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype, device_map=device_map)
        self._model.eval()
        self.tokenizer = HFTokenizer(self._processor.tokenizer, model_id)

    @staticmethod
    def _to_chat(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        chat = []
        for message in messages:
            if message.images:
                content: Any = [{"type": "image", "image": str(path)} for path in message.images]
                content.append({"type": "text", "text": message.content})
            else:
                content = [{"type": "text", "text": message.content}]
            chat.append({"role": message.role, "content": content})
        return chat

    def generate(
        self,
        messages: Sequence[ChatMessage],
        config: GenerationConfig,
        *,
        task: str,
        hints: Mapping[str, Any] | None = None,
    ) -> Generation:
        del task, hints  # the real model only sees the chat
        chat = self._to_chat(messages)
        text = self._processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(chat)
        inputs = self._processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self._model.device)
        kwargs: dict[str, Any] = {"max_new_tokens": config.max_new_tokens, "do_sample": config.temperature > 0}
        if config.temperature > 0:
            kwargs.update(temperature=config.temperature, top_p=config.top_p)
        if config.seed is not None:
            self._torch.manual_seed(config.seed)
        start = time.perf_counter()
        with self._torch.inference_mode():
            output = self._model.generate(**inputs, **kwargs)
        elapsed = time.perf_counter() - start
        new_tokens = output[0][inputs.input_ids.shape[1]:]
        decoded = self._processor.batch_decode([new_tokens], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return Generation(decoded, elapsed, int(new_tokens.shape[0]))
