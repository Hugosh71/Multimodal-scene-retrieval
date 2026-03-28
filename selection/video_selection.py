from __future__ import annotations

import math
from typing import List, Dict, Optional

from sentence_transformers import CrossEncoder

from config import CROSS_ENCODER_MODEL, TOP_K, RELEVANCE_THRESHOLD

_cross_encoder: Optional[CrossEncoder] = None


def _load_model() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print(f"  Loading CrossEncoder ({CROSS_ENCODER_MODEL}) …")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def select_scenes(user_request: str, candidate_scenes: List[Dict]) -> List[Dict]:
    if not candidate_scenes:
        return []

    model = _load_model()

    pairs = [(user_request, s.get("description", "")) for s in candidate_scenes]
    raw_scores = model.predict(pairs)

    def _sigmoid_score(x: float) -> float:
        return round(10 / (1 + math.exp(-x / 3)), 2)

    ranked = []
    for scene, raw_score in zip(candidate_scenes, raw_scores):
        raw = float(raw_score)
        if raw < RELEVANCE_THRESHOLD:
            continue
        ranked.append({**scene, "_raw": raw, "relevance_score": _sigmoid_score(raw)})

    if not ranked:
        print("  No scenes matched the query above the relevance threshold.")
        print(f"  (threshold = {RELEVANCE_THRESHOLD}; lower it in config.py to be more lenient)")
        return []

    ranked.sort(key=lambda x: x["_raw"], reverse=True)
    for i, item in enumerate(ranked, start=1):
        item["rank"] = i
        del item["_raw"]

    return ranked
