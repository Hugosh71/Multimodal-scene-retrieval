import os

FLORENCE2_MODEL_NAME = "microsoft/Florence-2-base"
FLORENCE2_DEVICE = "cpu"

N_SAMPLE_FRAMES = 12

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RELEVANCE_THRESHOLD = -3.0

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
CHROMA_COLLECTION_NAME = "scene_descriptions"

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "data", "videos")
FRAMES_DIR = os.path.join(os.path.dirname(__file__), "data", "frames")

SCENE_THRESHOLD = 27.0
MIN_SCENE_LEN = 15

FRAME_POSITION = 0.5

TOP_K = 5
