from __future__ import annotations

import argparse
from tqdm import tqdm

from config import TOP_K
from database.video_db import VideoDatabase
from database.vector_store import VectorStore
from preprocessing.scene_detection import detect_scenes
from preprocessing.frame_collection import collect_frames
from preprocessing.frame_description import describe_frames
from retrieval.rag import retrieve_scenes
from selection.video_selection import select_scenes


def ingest(force: bool = False) -> None:
    video_db = VideoDatabase()
    vector_store = VectorStore()

    videos = video_db.unprocessed_videos() if not force else video_db.list_videos()

    if not videos:
        print("No new videos to process. Add MP4/MKV/AVI files to data/videos/")
        return

    print(f"Found {len(videos)} video(s) to process.\n")

    for video_path in tqdm(videos, desc="Videos", unit="video"):
        print(f"\n{'='*60}")
        print(f"Processing: {video_path}")

        print("  [1/3] Detecting scenes …")
        scenes = detect_scenes(video_path)
        print(f"        {len(scenes)} scene(s) detected.")

        print("  [2/3] Collecting representative frames …")
        scenes = collect_frames(scenes)

        print("  [3/3] Describing frames with Florence-2 …")
        scenes = describe_frames(scenes)

        video_db.save_scenes(video_path, scenes)
        vector_store.add_scenes(scenes)

        print(f"        Indexed {len(scenes)} scene(s) into ChromaDB.")

    print(f"\nIngestion complete. Total indexed scenes: {vector_store.count()}")


def search(user_input: str, top_k: int = TOP_K) -> list:
    vector_store = VectorStore()

    print("Retrieving candidate scenes …")
    optimised_query, candidates = retrieve_scenes(
        user_input, vector_store=vector_store, top_k=top_k * 2
    )
    print(f"Optimised query: {optimised_query}")
    print(f"Candidates retrieved: {len(candidates)}\n")

    print("Re-ranking with CrossEncoder …")
    results = select_scenes(user_input, candidates)

    return results[:top_k]


def inspect(video_filter: str | None = None) -> None:
    video_db = VideoDatabase()
    all_scenes = video_db.all_scenes()

    if not all_scenes:
        print("No scenes indexed yet. Run: python main.py ingest")
        return

    if video_filter:
        all_scenes = [
            s for s in all_scenes
            if video_filter.lower() in s.get("video_path", "").lower()
        ]
        if not all_scenes:
            print(f"No scenes found matching video filter '{video_filter}'.")
            return

    from collections import defaultdict
    by_video: dict = defaultdict(list)
    for s in all_scenes:
        by_video[s.get("video_path", "unknown")].append(s)

    total = 0
    for video_path, scenes in by_video.items():
        print(f"\n{'='*70}")
        print(f"VIDEO: {video_path}")
        print(f"{'='*70}")
        scenes.sort(key=lambda x: x.get("start_time", 0))
        for s in scenes:
            total += 1
            print(f"\n  Scene : {s['scene_id']}")
            print(f"  Time  : {s.get('start_time', '?'):.1f}s – {s.get('end_time', '?'):.1f}s")
            print(f"  Frame : {s.get('frame_path', 'N/A')}")
            print(f"  Florence-2 : {s.get('frame_description', '').strip() or '(empty)'}")
            print(f"  {'-'*66}")

    print(f"\nTotal: {total} scene(s) across {len(by_video)} video(s).")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal Scene Search powered by LLMs")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_cmd = sub.add_parser("ingest", help="Index videos into the vector database")
    ingest_cmd.add_argument("--force", action="store_true", help="Re-process already indexed videos")

    search_cmd = sub.add_parser("search", help="Search for scenes matching a query")
    search_cmd.add_argument("query", help="Text prompt or path to a reference image")
    search_cmd.add_argument("--top-k", type=int, default=TOP_K, help=f"Number of results (default: {TOP_K})")

    inspect_cmd = sub.add_parser("inspect", help="Show stored descriptions for all indexed scenes")
    inspect_cmd.add_argument("--video", default=None, metavar="NAME", help="Filter by video filename substring")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.command == "ingest":
        ingest(force=args.force)

    elif args.command == "search":
        results = search(args.query, top_k=args.top_k)
        print(f"\nTop-{len(results)} scenes:\n")
        for r in results:
            print(f"  [{r.get('rank', '?')}] {r['scene_id']}  (score: {r.get('relevance_score', 'N/A')})")
            print(f"      Video: {r.get('video_path', 'N/A')}")
            print(f"      Time : {r.get('start_time', '?')}s – {r.get('end_time', '?')}s\n")

    elif args.command == "inspect":
        inspect(video_filter=args.video)
