# Multimodal Video Scene Search

Search video archives using natural language. Describe a scene in plain text (or provide a reference image) and the system returns the most relevant video segments.

Built entirely with open-source models.

---

## How it works

```
Video files
    │
    ▼
[Scene Detection]      PySceneDetect splits the video into scenes
    │
    ▼
[Frame Extraction]     One representative frame extracted per scene
    │
    ▼
[Florence-2]           Each frame described in detail (Image → Text)
    │
    ▼
[ChromaDB]             Descriptions embedded and indexed for vector search
    │
    ▼
[Search query]  ──────────────────────────────────────────────┐
    │                                                          │
    ▼                                                          ▼
[ChromaDB retrieval]                                   [Florence-2]
Top-k candidates                               (if query is an image)
    │
    ▼
[CrossEncoder]         Re-ranks candidates by relevance
    │
    ▼
[Results]              Ranked scenes with timestamps and scores
```

---

## Models used

| Model | Role | Size |
|---|---|---|
| `microsoft/Florence-2-base` | Frame captioning (Image → Text) | ~900 MB |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Result re-ranking | ~50 MB |
| `sentence-transformers` (ChromaDB default) | Text embeddings | ~90 MB |

All models are downloaded automatically on first run.

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

> **Dependency note:** Florence-2 requires `transformers<5.0.0` and `tokenizers<0.20.0`.
> If you encounter a `flash_attn` error, see the Troubleshooting section below.

---

## Usage

### 1. Add videos

Place your `.mp4`, `.mkv`, or `.avi` files in `data/videos/`.

### 2. Index the video library

```bash
python main.py ingest
```

Add `--force` to re-process already indexed videos:

```bash
python main.py ingest --force
```

### 3. Search

```bash
python main.py search "a woman holding a cigarette in a casino"
```

Search using a reference image:

```bash
python main.py search "path/to/reference_image.jpg"
```

Control the number of results (default: 5):

```bash
python main.py search "a car chase at night" --top-k 3
```

### 4. Inspect stored descriptions

```bash
python main.py inspect
```

Filter by video:

```bash
python main.py inspect --video skyfall
```

---

## Configuration

All settings are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `FLORENCE2_DEVICE` | `cpu` | Set to `cuda` if you have a GPU |
| `SCENE_THRESHOLD` | `27.0` | Lower = more scenes detected |
| `N_SAMPLE_FRAMES` | `12` | Frames sampled per scene |
| `TOP_K` | `5` | Number of results returned |
| `RELEVANCE_THRESHOLD` | `-3.0` | Lower = more lenient matching |

---

## Project structure

```
├── main.py                     # CLI entry point
├── config.py                   # All configuration
├── requirements.txt
│
├── preprocessing/
│   ├── scene_detection.py      # PySceneDetect wrapper
│   ├── frame_collection.py     # Frame extraction
│   └── frame_description.py   # Florence-2 captioning
│
├── database/
│   ├── vector_store.py         # ChromaDB wrapper
│   └── video_db.py             # Scene metadata catalog
│
├── retrieval/
│   ├── prompt_engineering.py   # Query building (text or image)
│   └── rag.py                  # Vector retrieval pipeline
│
├── selection/
│   └── video_selection.py      # CrossEncoder re-ranking
│
└── data/
    └── videos/                 # Place your video files here
```

---

## Troubleshooting

**`flash_attn` ImportError**

Florence-2's model file imports `flash_attn` which requires CUDA build tools. Patch the cached model file to make the import optional:

1. Find the file:
   ```
   ~/.cache/huggingface/hub/models--microsoft--Florence-2-base/snapshots/<hash>/modeling_florence2.py
   ```
2. Wrap both `if is_flash_attn_2_available():` import blocks in `try/except ImportError: pass`

**`tokenizers` version conflict**

```bash
pip install "transformers>=4.41.0,<5.0.0" "tokenizers>=0.15.0,<0.20.0"
```
