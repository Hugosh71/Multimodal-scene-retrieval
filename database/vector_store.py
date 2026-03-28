from __future__ import annotations

from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, TOP_K


class VectorStore:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._ef = embedding_functions.DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_scenes(self, scenes: List[Dict]) -> None:
        ids, documents, metadatas = [], [], []

        for scene in scenes:
            frame_desc = scene.get("frame_description", "")
            video_desc = scene.get("video_description", "")
            useful_pegasus = video_desc if len(video_desc.strip()) > 20 else ""
            description = f"{frame_desc} {useful_pegasus}".strip() if useful_pegasus else frame_desc
            if not description:
                continue

            meta = {
                k: str(v)
                for k, v in scene.items()
                if k not in ("scene_id", "frame_description", "video_description")
                and v is not None
            }

            ids.append(scene["scene_id"])
            documents.append(description)
            metadatas.append(meta)

        if ids:
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, text: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        results = self._collection.query(
            query_texts=[text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        scenes: List[Dict] = []
        for i, scene_id in enumerate(results["ids"][0]):
            scenes.append(
                {
                    "scene_id": scene_id,
                    "description": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                    **results["metadatas"][0][i],
                }
            )

        return scenes

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(CHROMA_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
