"""Model boundary. Mock mode is reproducible; Qwen loading is optional."""

import json
import time
from pathlib import Path
from typing import Protocol

from transmitter.dataset.schema import ImageAttributes


class VisionExtractor(Protocol):
    model_name: str

    def extract(self, image_path: Path, expected: ImageAttributes | None = None) -> tuple[ImageAttributes, float]: ...


class MockExtractor:
    model_name = "mock-ground-truth-v1"

    def extract(self, image_path: Path, expected: ImageAttributes | None = None) -> tuple[ImageAttributes, float]:
        if expected is None:
            raise ValueError("MockExtractor requires expected attributes")
        return expected, 0.0


class QwenVisionExtractor:
    """Qwen2.5-VL adapter returning validated image attributes."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> None:
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
        except ImportError as error:
            raise ImportError("Install the 'model' extra to use QwenVisionExtractor") from error

        self.model_name = model_id
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        self._process_vision_info = process_vision_info

    def extract(self, image_path: Path, expected: ImageAttributes | None = None) -> tuple[ImageAttributes, float]:
        prompt = Path(__file__).parents[1].parent / "prompts" / "visual_extraction.txt"
        messages = [{"role": "user", "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": prompt.read_text(encoding="utf-8")}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        start = time.perf_counter()
        generated_ids = self._model.generate(**inputs, max_new_tokens=256, do_sample=False)
        elapsed = time.perf_counter() - start
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        response = self._processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        try:
            data = json.loads(response)
            attributes = ImageAttributes.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Qwen returned invalid attribute JSON: {response!r}") from error
        return attributes, elapsed
