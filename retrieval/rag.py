from __future__ import annotations

from typing import List, Dict

from config import TOP_K
from database.vector_store import VectorStore
from retrieval.prompt_engineering import build_query


def retrieve_scenes(
    user_input: str,
    vector_store: VectorStore | None = None,
    top_k: int = TOP_K,
) -> tuple[str, List[Dict]]:
    if vector_store is None:
        vector_store = VectorStore()

    optimised_query = build_query(user_input)
    candidate_scenes = vector_store.query(optimised_query, top_k=top_k)

    return optimised_query, candidate_scenes
