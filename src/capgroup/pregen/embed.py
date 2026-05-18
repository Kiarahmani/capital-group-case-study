"""Embed train posts (documents) and test article bodies (queries) via Voyage."""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import voyageai
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_XLSX = REPO_ROOT / "data" / "inputs" / "train.xlsx"
ARTICLES_DIR = REPO_ROOT / "data" / "cache" / "articles"
CONFIG_PATH = REPO_ROOT / "config.yaml"
EMBED_NPZ = REPO_ROOT / "data" / "cache" / "embeddings.npz"
EMBED_META = REPO_ROOT / "data" / "cache" / "embeddings_meta.json"

URL_RE = re.compile(r"https?://\S+|bit\.ly/\S+")
DISCLOSURE_RE = re.compile(r"important disclosures?:?\s*", re.IGNORECASE)


def clean_for_embedding(text: str) -> str:
    text = URL_RE.sub("", text)
    text = DISCLOSURE_RE.sub("", text)
    return " ".join(text.split())


def classify_track(url: str, tracks: dict) -> str:
    for name, pattern in tracks.items():
        if pattern in url:
            return name
    return "other"


def load_train(tracks: dict) -> list[dict]:
    df = pd.read_excel(TRAIN_XLSX, sheet_name="nearShoreTrain")
    return [
        {
            "source": "train",
            "postId": int(row.postId),
            "url": row.URL,
            "audience_track": classify_track(row.URL, tracks),
            "text": clean_for_embedding(row.postOriginal),
        }
        for row in df.itertuples()
    ]


def load_test() -> list[dict]:
    records = []
    for path in sorted(ARTICLES_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        rec = json.loads(path.read_text())
        text = rec["body"] if rec["body"] else rec.get("title", "")
        records.append({
            "source": "test",
            "postId": rec["postId"],
            "url": rec["url"],
            "audience_track": rec["audience_track"],
            "title": rec.get("title", ""),
            "text": text,
        })
    records.sort(key=lambda r: r["postId"])
    return records


def embed_batch(client, texts: list[str], input_type: str, model: str) -> np.ndarray:
    BATCH = 128
    out = []
    for i in range(0, len(texts), BATCH):
        result = client.embed(
            texts=texts[i:i + BATCH],
            model=model,
            input_type=input_type,
        )
        out.extend(result.embeddings)
    return np.array(out, dtype=np.float32)


def main():
    load_dotenv(REPO_ROOT / ".env")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    model = config["models"]["embedding"]
    tracks = config["audience_tracks"]

    print("Loading corpora...")
    train = load_train(tracks)
    test = load_test()
    print(f"  train: {len(train)} posts")
    print(f"  test:  {len(test)} articles")

    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    print(f"Embedding train as 'document' (model={model})...")
    train_vecs = embed_batch(client, [r["text"] for r in train], "document", model)
    print(f"  shape: {train_vecs.shape}")

    print(f"Embedding test as 'query' (model={model})...")
    test_vecs = embed_batch(client, [r["text"] for r in test], "query", model)
    print(f"  shape: {test_vecs.shape}")

    np.savez_compressed(EMBED_NPZ, train=train_vecs, test=test_vecs)
    meta = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model,
        "dim": int(train_vecs.shape[1]),
        "train_count": len(train),
        "test_count": len(test),
        "train": [{k: v for k, v in r.items() if k != "text"} for r in train],
        "test": [{k: v for k, v in r.items() if k != "text"} for r in test],
    }
    EMBED_META.write_text(json.dumps(meta, indent=2))
    size_kb = EMBED_NPZ.stat().st_size // 1024
    print(f"Saved {EMBED_NPZ.name} ({size_kb} KB) and {EMBED_META.name}")


if __name__ == "__main__":
    main()
