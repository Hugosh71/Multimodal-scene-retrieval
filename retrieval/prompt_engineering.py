from __future__ import annotations

from pathlib import Path

from PIL import Image

from preprocessing.frame_description import caption_image

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def build_query(user_input: str) -> str:
    if Path(user_input).suffix.lower() in _IMAGE_EXTENSIONS and Path(user_input).exists():
        image = Image.open(user_input).convert("RGB")
        return caption_image(image, task="<DETAILED_CAPTION>")
    return user_input
