"""
Video Description  (Video to Text)
------------------------------------
Generates an enriched, temporally-aware description for each scene by:

  1. Sampling N_SAMPLE_FRAMES frames uniformly across the scene clip.
  2. Captioning each sampled frame with Florence-2 <CAPTION> (reuses the
     shared model loaded by frame_description.py – no extra RAM cost).
  3. Feeding the per-frame captions + the main frame_description through
     Pegasus to produce a single coherent ``video_description``.

All models are open-source and run locally.
"""
from __future__ import annotations

import os
from typing import List, Dict

import cv2
from PIL import Image as PILImage
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

from preprocessing.frame_description import caption_image

from config import (
    PEGASUS_MODEL_NAME,
    PEGASUS_MAX_LENGTH,
    PEGASUS_DEVICE,
    FRAMES_DIR,
    N_SAMPLE_FRAMES,
)


# ── Lazy model loading ─────────────────────────────────────────────────────

_tokenizer: PegasusTokenizer | None = None
_model: PegasusForConditionalGeneration | None = None


def _load_model() -> tuple:
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = PegasusTokenizer.from_pretrained(PEGASUS_MODEL_NAME)
        _model = PegasusForConditionalGeneration.from_pretrained(PEGASUS_MODEL_NAME)
        _model.to(PEGASUS_DEVICE)
        _model.eval()
    return _tokenizer, _model


# ── Frame sampling helpers ─────────────────────────────────────────────────

def _sample_frame_indices(start: int, end: int, n: int) -> List[int]:
    """Return *n* frame indices uniformly distributed between *start* and *end*."""
    if end <= start:
        return [start]
    return [int(start + i * (end - start) / max(n - 1, 1)) for i in range(n)]


def _frame_to_caption(bgr_frame) -> str:
    """Caption a single OpenCV BGR frame using Florence-2 <CAPTION>."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil_image = PILImage.fromarray(rgb)
    return caption_image(pil_image, task="<CAPTION>")


# ── Public API ─────────────────────────────────────────────────────────────

def describe_video(scene: Dict) -> Dict:
    """
    Generate an enriched ``video_description`` for *scene* using Pegasus.

    Parameters
    ----------
    scene : Dict
        Must contain ``"video_path"``, ``"start_frame"``, ``"end_frame"``,
        and ideally ``"frame_description"``.

    Returns
    -------
    Dict
        Same dict with ``"video_description"`` added.
    """
    tokenizer, model = _load_model()

    # ── Sample frames and caption each with Florence-2 ─────────────────────
    indices = _sample_frame_indices(
        scene["start_frame"], scene["end_frame"], N_SAMPLE_FRAMES
    )

    cap = cv2.VideoCapture(scene["video_path"])
    frame_captions: List[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_captions.append(_frame_to_caption(frame))
    cap.release()

    # ── Build input text ───────────────────────────────────────────────────
    temporal_context = " | ".join(frame_captions)
    base_description = scene.get("frame_description", "")

    input_text = (
        f"Scene description: {base_description}. "
        f"Temporal visual context: {temporal_context}. "
        "Provide an enriched, coherent description of this video scene."
    )

    # ── Run Pegasus ────────────────────────────────────────────────────────
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    ).to(PEGASUS_DEVICE)

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=PEGASUS_MAX_LENGTH,
            num_beams=4,
            early_stopping=True,
        )

    video_description = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    scene["video_description"] = video_description
    return scene


def describe_videos(scenes: List[Dict]) -> List[Dict]:
    """Batch wrapper – enriches every scene in *scenes*."""
    for scene in scenes:
        describe_video(scene)
    return scenes


if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 5:
        print("Usage: python video_description.py <video_path> <start_frame> <end_frame> <scene_id>")
        sys.exit(1)

    dummy = {
        "video_path": sys.argv[1],
        "start_frame": int(sys.argv[2]),
        "end_frame": int(sys.argv[3]),
        "scene_id": sys.argv[4],
        "frame_description": "",
    }
    result = describe_video(dummy)
    print(result["video_description"])
