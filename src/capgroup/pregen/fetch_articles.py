"""Fetch and cache article HTML bodies for the 16 test posts."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import trafilatura
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config.yaml"
TEST_PATH = REPO_ROOT / "data" / "inputs" / "test.xlsx"
TEST_SHEET = "Sheet1"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "articles"
INDEX_PATH = CACHE_DIR / "_index.json"

USER_AGENT = "Mozilla/5.0 (compatible; CapGroupCaseStudy/1.0)"
TIMEOUT_SECONDS = 30
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.0
INTER_REQUEST_SLEEP_SECONDS = 0.5
SHORT_BODY_THRESHOLD = 200


def load_tracks() -> dict[str, str]:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)["audience_tracks"]


def extract_track(url: str, tracks: dict[str, str]) -> str:
    if not isinstance(url, str):
        return "other"
    for name, fragment in tracks.items():
        if fragment in url:
            return name
    return "other"


def slug_to_title(url: str) -> str:
    path = urlparse(url).path
    last = path.rstrip("/").rsplit("/", 1)[-1]
    if last.endswith(".html"):
        last = last[: -len(".html")]
    return last.replace("-", " ").strip().title()


def http_get(url: str) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=TIMEOUT_SECONDS,
            )
            if 500 <= resp.status_code < 600 and attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)
                continue
            return resp
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("http_get reached an unreachable state")


def try_fetch(url: str) -> tuple[int | None, str | None]:
    try:
        resp = http_get(url)
        return resp.status_code, resp.text
    except (requests.Timeout, requests.ConnectionError, requests.RequestException):
        return None, None


def extract_body_and_title(html: str) -> tuple[str | None, str | None]:
    body = trafilatura.extract(html)
    title: str | None = None
    try:
        meta = trafilatura.extract_metadata(html)
        if meta is not None:
            title = getattr(meta, "title", None)
    except Exception:
        title = None
    return body, title


def fetch_one(post_id: int, url: str, tracks: dict[str, str]) -> dict:
    track = extract_track(url, tracks)
    http_status, html = try_fetch(url)

    # postId 96 has a query string; if the first request 4xx'd, retry without it.
    if (http_status is not None and 400 <= http_status < 500 and "?" in url):
        bare_url = url.split("?", 1)[0]
        http_status, html = try_fetch(bare_url)

    body: str | None = None
    title: str | None = None
    if http_status == 200 and html:
        body, title = extract_body_and_title(html)

    fallback = (
        http_status is None
        or http_status >= 400
        or not body
    )

    if fallback:
        record_body = ""
        record_title = slug_to_title(url)
        fetch_status = "fallback"
    else:
        record_body = body or ""
        record_title = title or slug_to_title(url)
        fetch_status = "ok"

    return {
        "postId": int(post_id),
        "url": url,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetch_status": fetch_status,
        "http_status": http_status,
        "title": record_title,
        "body": record_body,
        "char_count": len(record_body),
        "audience_track": track,
    }


def main() -> None:
    tracks = load_tracks()
    df = pd.read_excel(TEST_PATH, sheet_name=TEST_SHEET)
    pairs = [(int(row["postId"]), str(row["URL"])) for _, row in df.iterrows()]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    ok_count = 0
    fallback_count = 0

    for i, (post_id, url) in enumerate(tqdm(pairs, desc="fetch", unit="article")):
        record = fetch_one(post_id, url, tracks)
        out_path = CACHE_DIR / f"{post_id}.json"
        with out_path.open("w") as f:
            json.dump(record, f, indent=2)

        if record["fetch_status"] == "ok":
            ok_count += 1
        else:
            fallback_count += 1

        if record["fetch_status"] == "ok" and record["char_count"] < SHORT_BODY_THRESHOLD:
            print(
                f"postId={post_id} status=ok http={record['http_status']} "
                f"chars={record['char_count']} (short body)"
            )
        else:
            print(
                f"postId={post_id} status={record['fetch_status']} "
                f"http={record['http_status']} chars={record['char_count']}"
            )

        summaries.append({
            "postId": record["postId"],
            "fetch_status": record["fetch_status"],
            "char_count": record["char_count"],
            "audience_track": record["audience_track"],
        })

        if i < len(pairs) - 1:
            time.sleep(INTER_REQUEST_SLEEP_SECONDS)

    index = {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": len(pairs),
        "ok": ok_count,
        "fallback": fallback_count,
        "articles": summaries,
    }
    with INDEX_PATH.open("w") as f:
        json.dump(index, f, indent=2)

    print(f"fetched: ok={ok_count} fallback={fallback_count}")


if __name__ == "__main__":
    main()
