"""Stratified retrieval over Voyage embeddings: same-track filter + dedupe-by-URL."""

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_XLSX = REPO_ROOT / "data" / "inputs" / "train.xlsx"
EMBED_NPZ = REPO_ROOT / "data" / "cache" / "embeddings.npz"
EMBED_META = REPO_ROOT / "data" / "cache" / "embeddings_meta.json"

# Clean retrieved examples before they go into the prompt.
DISCLOSURE_TRAILER_RE = re.compile(
    r"\s*important disclosures?:?\s*\S*\s*$", re.IGNORECASE
)
EXCEL_NEWLINE_RE = re.compile(r"_x000D_")


def clean_example(text: str) -> str:
    text = EXCEL_NEWLINE_RE.sub("", text)
    text = DISCLOSURE_TRAILER_RE.sub("", text)
    return text.strip()


class Retriever:
    def __init__(self):
        arrays = np.load(EMBED_NPZ)
        self.train_vecs = arrays["train"]
        self.test_vecs = arrays["test"]
        self.meta = json.loads(EMBED_META.read_text())

        df = pd.read_excel(TRAIN_XLSX, sheet_name="nearShoreTrain")
        self.train_text_by_postid = dict(zip(df.postId, df.postOriginal))

        norms_train = np.linalg.norm(self.train_vecs, axis=1, keepdims=True)
        self.train_norm = self.train_vecs / norms_train
        norms_test = np.linalg.norm(self.test_vecs, axis=1, keepdims=True)
        self.test_norm = self.test_vecs / norms_test

        self.test_idx_by_postid = {
            r["postId"]: i for i, r in enumerate(self.meta["test"])
        }

    def retrieve(
        self,
        test_post_id: int,
        k: int = 8,
        same_track_only: bool = True,
        dedupe_by_url: bool = True,
        exclude_urls: Optional[set[str]] = None,
    ) -> list[dict]:
        test_idx = self.test_idx_by_postid[test_post_id]
        return self.retrieve_by_vector(
            query_vec=self.test_norm[test_idx],
            query_track=self.meta["test"][test_idx]["audience_track"],
            k=k,
            same_track_only=same_track_only,
            dedupe_by_url=dedupe_by_url,
            exclude_urls=exclude_urls,
        )

    def retrieve_by_vector(
        self,
        query_vec: np.ndarray,
        query_track: str,
        k: int = 8,
        same_track_only: bool = True,
        dedupe_by_url: bool = True,
        exclude_urls: Optional[set[str]] = None,
    ) -> list[dict]:
        """Lower-level retrieval against an arbitrary query vector + track."""
        # Ensure query is normalized
        q = query_vec / np.linalg.norm(query_vec)
        sims = self.train_norm @ q
        exclude_urls = exclude_urls or set()

        candidates = []
        for i, train_meta in enumerate(self.meta["train"]):
            if same_track_only and train_meta["audience_track"] != query_track:
                continue
            if train_meta["url"] in exclude_urls:
                continue
            candidates.append((float(sims[i]), i, train_meta))
        candidates.sort(key=lambda x: -x[0])

        results = []
        seen_urls = set()
        for sim, i, m in candidates:
            if dedupe_by_url and m["url"] in seen_urls:
                continue
            seen_urls.add(m["url"])
            results.append(
                {
                    "similarity": sim,
                    "train_post_id": m["postId"],
                    "url": m["url"],
                    "audience_track": m["audience_track"],
                    "text_raw": self.train_text_by_postid[m["postId"]],
                    "text_clean": clean_example(self.train_text_by_postid[m["postId"]]),
                }
            )
            if len(results) >= k:
                break
        return results
