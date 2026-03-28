from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from config import SCENE_THRESHOLD, MIN_SCENE_LEN


def detect_scenes(video_path: str) -> List[Dict]:
    video_path = str(Path(video_path).resolve())
    video_stem = Path(video_path).stem

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=SCENE_THRESHOLD, min_scene_len=MIN_SCENE_LEN)
    )

    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    scenes: List[Dict] = []
    for idx, (start_tc, end_tc) in enumerate(scene_list):
        scenes.append(
            {
                "scene_id": f"{video_stem}_scene_{idx:04d}",
                "video_path": video_path,
                "start_frame": start_tc.get_frames(),
                "end_frame": end_tc.get_frames(),
                "start_time": start_tc.get_seconds(),
                "end_time": end_tc.get_seconds(),
            }
        )

    return scenes
