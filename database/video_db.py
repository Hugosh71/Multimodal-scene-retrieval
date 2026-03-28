from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Optional

from config import VIDEO_DIR

_SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
_CATALOG_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "video_catalog.json")


class VideoDatabase:
    def __init__(self) -> None:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        self._catalog: Dict[str, Dict] = self._load_catalog()

    def _load_catalog(self) -> Dict[str, Dict]:
        if os.path.exists(_CATALOG_FILE):
            with open(_CATALOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_catalog(self) -> None:
        os.makedirs(os.path.dirname(_CATALOG_FILE), exist_ok=True)
        with open(_CATALOG_FILE, "w", encoding="utf-8") as f:
            json.dump(self._catalog, f, indent=2, ensure_ascii=False)

    def list_videos(self) -> List[str]:
        return [
            str(p.resolve())
            for p in Path(VIDEO_DIR).rglob("*")
            if p.suffix.lower() in _SUPPORTED_EXTENSIONS
        ]

    def unprocessed_videos(self) -> List[str]:
        all_videos = self.list_videos()
        return [v for v in all_videos if v not in self._catalog]

    def save_scenes(self, video_path: str, scenes: List[Dict]) -> None:
        self._catalog[video_path] = {"scenes": scenes}
        self._save_catalog()

    def load_scenes(self, video_path: str) -> Optional[List[Dict]]:
        entry = self._catalog.get(video_path)
        if entry:
            return entry.get("scenes")
        return None

    def all_scenes(self) -> List[Dict]:
        scenes: List[Dict] = []
        for entry in self._catalog.values():
            scenes.extend(entry.get("scenes", []))
        return scenes
