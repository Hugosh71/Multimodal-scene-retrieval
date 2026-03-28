from __future__ import annotations

import os
from typing import List, Dict

import cv2

from config import FRAMES_DIR, FRAME_POSITION


def collect_frames(scenes: List[Dict]) -> List[Dict]:
    os.makedirs(FRAMES_DIR, exist_ok=True)

    for scene in scenes:
        target_frame = int(
            scene["start_frame"]
            + FRAME_POSITION * (scene["end_frame"] - scene["start_frame"])
        )

        cap = cv2.VideoCapture(scene["video_path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            scene["frame_path"] = None
            continue

        frame_path = os.path.join(FRAMES_DIR, f"{scene['scene_id']}.jpg")
        cv2.imwrite(frame_path, frame)
        scene["frame_path"] = frame_path

    return scenes
