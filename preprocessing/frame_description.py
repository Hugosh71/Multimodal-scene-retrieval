from __future__ import annotations

from typing import List, Dict

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from config import FLORENCE2_MODEL_NAME, FLORENCE2_DEVICE

_processor: AutoProcessor | None = None
_model: AutoModelForCausalLM | None = None


def _load_model() -> tuple:
    global _processor, _model
    if _processor is None or _model is None:
        print(f"  Loading Florence-2 ({FLORENCE2_MODEL_NAME}) …")
        _processor = AutoProcessor.from_pretrained(
            FLORENCE2_MODEL_NAME, trust_remote_code=True
        )
        _model = AutoModelForCausalLM.from_pretrained(
            FLORENCE2_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        _model.to(FLORENCE2_DEVICE)
        _model.eval()
    return _processor, _model


def caption_image(image: Image.Image, task: str = "<MORE_DETAILED_CAPTION>") -> str:
    processor, model = _load_model()

    inputs = processor(text=task, images=image, return_tensors="pt")
    inputs = {k: v.to(FLORENCE2_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            num_beams=3,
            do_sample=False,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=task, image_size=(image.width, image.height)
    )
    return parsed[task].strip()


def describe_frame(scene: Dict) -> Dict:
    if not scene.get("frame_path"):
        scene["frame_description"] = ""
        return scene

    image = Image.open(scene["frame_path"]).convert("RGB")
    scene["frame_description"] = caption_image(image, task="<MORE_DETAILED_CAPTION>")
    return scene


def describe_frames(scenes: List[Dict]) -> List[Dict]:
    _load_model()
    for scene in scenes:
        describe_frame(scene)
    return scenes
