"""Fetch and cache article HTML bodies for the 16 test posts.

Source strategy per article:
  A. Wayback Machine via direct /web/<year>/<url> redirect. Try year prefixes
     in order (2024, 2023, 2025) — the first snapshot Wayback redirects us to
     that yields a real article body wins. We avoid the CDX search API because
     it is aggressively rate-limited.

     If all year prefixes return auth-wall / no-body AND the URL is the
     AEM-internal form (/content/capital-group/us/en/<segment>/home/insights/
     articles/...), canonicalize it to the user-facing form
     (/<segment>/insights/articles/...) and retry the year-fallback.
  B. Live capitalgroup.com URL (fallback — kept in case Capital Group
     un-deprecates).
  C. Slug-derived title with empty body (last resort).
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import trafilatura
import yaml


def log(msg: str) -> None:
    print(msg, flush=True, file=sys.stderr)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config.yaml"
TEST_PATH = REPO_ROOT / "data" / "inputs" / "test.xlsx"
TEST_SHEET = "Sheet1"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "articles"
INDEX_PATH = CACHE_DIR / "_index.json"

USER_AGENT = "Mozilla/5.0 (compatible; CapGroupCaseStudy/1.0; +case study research)"
TIMEOUT_SECONDS = 15
MAX_RETRIES = 1
RETRY_BACKOFF_SECONDS = 1.5
INTER_REQUEST_SLEEP_SECONDS = 1.5
MIN_BODY_CHARS = 200
SHORT_BODY_THRESHOLD = 200

# Wayback /web/<year>/<url> year prefixes tried in order. First snapshot that
# yields a real body wins. Articles span 2022-2024 vintage; 2023 snapshots are
# most reliably pre-authwall.
WAYBACK_YEARS = ("2024", "2023", "2025")

AEM_PATTERN = re.compile(
    r"^https?://www\.capitalgroup\.com"
    r"/content/capital-group/us/en/([^/]+)/home/insights/articles/(.+)$"
)

WAYBACK_FAILURE_MARKERS = (
    "wayback-toolbar",
    "Internet Archive",
    "Sorry, the page is not available",
)


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


def strip_query(url: str) -> str:
    return url.split("?", 1)[0]


def canonicalize_aem_url(url: str) -> str | None:
    match = AEM_PATTERN.match(strip_query(url))
    if match is None:
        return None
    segment, rest = match.group(1), match.group(2)
    return f"https://www.capitalgroup.com/{segment}/insights/articles/{rest}"


def http_get(url: str, timeout: int = TIMEOUT_SECONDS) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
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


def body_looks_real(body: str | None) -> bool:
    if not body or len(body) < MIN_BODY_CHARS:
        return False
    for marker in WAYBACK_FAILURE_MARKERS:
        if marker in body:
            return False
    return True


def try_wayback_year(year: str, original_url: str) -> tuple[int | None, str | None, str | None, str | None]:
    """Try /web/<year>/<original_url>. Returns (http_status, body, title, final_url).

    `final_url` is the Wayback-redirected snapshot URL (e.g. /web/20240223175438/...)
    so we can record the actual snapshot timestamp picked by Wayback.
    """
    target = f"https://web.archive.org/web/{year}/{original_url}"
    try:
        resp = http_get(target)
    except (requests.Timeout, requests.ConnectionError, requests.RequestException):
        return None, None, None, None
    if resp.status_code != 200 or not resp.text:
        return resp.status_code, None, None, resp.url
    body, title = extract_body_and_title(resp.text)
    return resp.status_code, body, title, resp.url


def try_live(url: str) -> tuple[int | None, str | None, str | None]:
    try:
        resp = http_get(url)
    except (requests.Timeout, requests.ConnectionError, requests.RequestException):
        return None, None, None
    if resp.status_code != 200 or not resp.text:
        return resp.status_code, None, None
    body, title = extract_body_and_title(resp.text)
    return resp.status_code, body, title


def fetch_article_body(
    url: str,
    audience_track: str,
    post_id: int | None = None,
) -> dict:
    """Fetch an article body via Wayback year-fallback + trafilatura.

    Returns a record matching the per-article cache schema:
      postId, url, [canonical_url], source, fetched_at, fetch_status,
      http_status, title, body, char_count, audience_track,
      [wayback_year], [wayback_snapshot_url]

    `post_id` is included in the record if provided; otherwise omitted. The
    caller is responsible for caching the record on disk if desired.

    Strategy (same as the test-article pre-gen pipeline):
      1. Year-fallback against the original URL via Wayback (/web/<year>/<url>).
      2. If URL is AEM-internal, canonicalize and retry year-fallback.
      3. Live URL fallback.
      4. Slug-title fallback (body="").
    """
    log_pid = f"postId={post_id}" if post_id is not None else f"url={url[-60:]}"
    stripped = strip_query(url)
    t0 = time.monotonic()
    canonical_url: str | None = None
    last_http_status: int | None = None

    # Step 1: year-fallback against the original (query-stripped) URL.
    for year in WAYBACK_YEARS:
        log(f"  {log_pid} /web/{year}/...")
        status, body, title, final_url = try_wayback_year(year, stripped)
        last_http_status = status
        if status == 200 and body_looks_real(body):
            log(f"  {log_pid} ok via wayback year={year} in {time.monotonic()-t0:.1f}s")
            return _record(
                post_id=post_id, url=url, canonical_url=None,
                source="wayback", fetch_status="ok", http_status=status,
                body=body or "", title=title or slug_to_title(url),
                track=audience_track, wayback_year=year, wayback_snapshot_url=final_url,
            )
        time.sleep(INTER_REQUEST_SLEEP_SECONDS)

    # Step 2: if URL is AEM-internal, canonicalize and retry year-fallback.
    canonical_candidate = canonicalize_aem_url(stripped)
    if canonical_candidate is not None:
        canonical_url = canonical_candidate
        for year in WAYBACK_YEARS:
            log(f"  {log_pid} canonical /web/{year}/...")
            status, body, title, final_url = try_wayback_year(year, canonical_candidate)
            last_http_status = status
            if status == 200 and body_looks_real(body):
                log(f"  {log_pid} ok via canonical wayback year={year} in {time.monotonic()-t0:.1f}s")
                return _record(
                    post_id=post_id, url=url, canonical_url=canonical_url,
                    source="wayback", fetch_status="ok", http_status=status,
                    body=body or "", title=title or slug_to_title(url),
                    track=audience_track, wayback_year=year, wayback_snapshot_url=final_url,
                )
            time.sleep(INTER_REQUEST_SLEEP_SECONDS)

    # Step 3: live URL fallback (use canonical if we derived one).
    live_target = canonical_url or url
    log(f"  {log_pid} live URL...")
    live_status, live_body, live_title = try_live(live_target)
    if live_status == 200 and body_looks_real(live_body):
        log(f"  {log_pid} ok via live in {time.monotonic()-t0:.1f}s")
        return _record(
            post_id=post_id, url=url, canonical_url=canonical_url,
            source="live", fetch_status="ok", http_status=live_status,
            body=live_body or "", title=live_title or slug_to_title(url),
            track=audience_track, wayback_year=None, wayback_snapshot_url=None,
        )

    # Step 4: slug-title fallback.
    log(f"  {log_pid} fallback in {time.monotonic()-t0:.1f}s")
    return _record(
        post_id=post_id, url=url, canonical_url=canonical_url,
        source="fallback", fetch_status="fallback",
        http_status=live_status if live_status is not None else last_http_status,
        body="", title=slug_to_title(url),
        track=audience_track, wayback_year=None, wayback_snapshot_url=None,
    )


def fetch_one(post_id: int, url: str, tracks: dict[str, str]) -> dict:
    """Backward-compatible wrapper used by the test-article pre-gen flow."""
    track = extract_track(url, tracks)
    return fetch_article_body(url=url, audience_track=track, post_id=post_id)


def _record(
    *,
    post_id: int | None,
    url: str,
    canonical_url: str | None,
    source: str,
    fetch_status: str,
    http_status: int | None,
    body: str,
    title: str,
    track: str,
    wayback_year: str | None,
    wayback_snapshot_url: str | None,
) -> dict:
    record: dict = {}
    if post_id is not None:
        record["postId"] = int(post_id)
    record["url"] = url
    if canonical_url is not None:
        record["canonical_url"] = canonical_url
    record.update({
        "source": source,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetch_status": fetch_status,
        "http_status": http_status,
        "title": title,
        "body": body,
        "char_count": len(body),
        "audience_track": track,
    })
    if source == "wayback":
        if wayback_year is not None:
            record["wayback_year"] = wayback_year
        if wayback_snapshot_url is not None:
            record["wayback_snapshot_url"] = wayback_snapshot_url
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and cache article bodies for the 16 test postIds. Wayback "
            "is non-deterministic; the committed cache pins the snapshots used "
            "in the submission run. Default behavior is to skip postIds whose "
            "cache file already exists — pass --force to overwrite."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch articles even when a cache file already exists.",
    )
    args = parser.parse_args()

    tracks = load_tracks()
    df = pd.read_excel(TEST_PATH, sheet_name=TEST_SHEET)
    pairs = [(int(row["postId"]), str(row["URL"])) for _, row in df.iterrows()]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    ok_count = 0
    fallback_count = 0
    skipped_count = 0
    fetched_any = False

    for i, (post_id, url) in enumerate(pairs):
        log(f"[{i+1}/{len(pairs)}] postId={post_id}")
        out_path = CACHE_DIR / f"{post_id}.json"
        is_cached = out_path.exists() and not args.force

        if is_cached:
            with out_path.open() as f:
                record = json.load(f)
            skipped_count += 1
            log(f"  → postId={post_id} SKIP (cache exists; --force to re-fetch)")
        else:
            record = fetch_one(post_id, url, tracks)
            with out_path.open("w") as f:
                json.dump(record, f, indent=2)
            fetched_any = True
            short = record["fetch_status"] == "ok" and record["char_count"] < SHORT_BODY_THRESHOLD
            log(
                f"  → postId={post_id} source={record['source']} "
                f"status={record['fetch_status']} chars={record['char_count']}"
                f"{' (short body)' if short else ''}"
            )

        if record["fetch_status"] == "ok":
            ok_count += 1
        else:
            fallback_count += 1

        summary = {
            "postId": record["postId"],
            "source": record["source"],
            "fetch_status": record["fetch_status"],
            "char_count": record["char_count"],
            "audience_track": record["audience_track"],
        }
        if "wayback_year" in record:
            summary["wayback_year"] = record["wayback_year"]
        if "canonical_url" in record:
            summary["canonical_url"] = record["canonical_url"]
        summaries.append(summary)

        # Only sleep between actual network requests, not between cache reads.
        if not is_cached and i < len(pairs) - 1:
            time.sleep(INTER_REQUEST_SLEEP_SECONDS)

    # Only rewrite the index if we actually fetched anything. Pure skip-all
    # runs leave the committed _index.json untouched.
    if fetched_any:
        index = {
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total": len(pairs),
            "ok": ok_count,
            "fallback": fallback_count,
            "articles": summaries,
        }
        with INDEX_PATH.open("w") as f:
            json.dump(index, f, indent=2)

    suffix = f" (skipped: {skipped_count})" if skipped_count else ""
    log(f"fetched: ok={ok_count} fallback={fallback_count}{suffix}")


if __name__ == "__main__":
    main()
