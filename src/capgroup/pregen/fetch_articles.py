"""Fetch and cache article HTML bodies for the 16 test posts.

Source strategy per article:
  A. Wayback Machine snapshot, selected via the CDX API. We enumerate
     all 200-status snapshots, filter out small responses (auth walls /
     redirects), and pick the earliest remaining one — earliest is most
     likely pre-deprecation / pre-authwall.

     If CDX returns nothing AND the URL is the AEM-internal form
     (/content/capital-group/us/en/<segment>/home/insights/articles/...),
     canonicalize it to the user-facing form
     (/<segment>/insights/articles/...) and retry CDX.
  B. Live capitalgroup.com URL (fallback — kept in case Capital Group
     un-deprecates).
  C. Slug-derived title with empty body (last resort).
"""

import json
import re
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

USER_AGENT = "Mozilla/5.0 (compatible; CapGroupCaseStudy/1.0; +case study research)"
TIMEOUT_SECONDS = 30
CDX_TIMEOUT_SECONDS = 20
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2.0
INTER_REQUEST_SLEEP_SECONDS = 1.5
MIN_BODY_CHARS = 200
SHORT_BODY_THRESHOLD = 200

# CDX snapshot selection thresholds.
CDX_API = "https://web.archive.org/cdx/search/cdx"
CDX_MIN_LENGTH = 10000  # bytes; smaller responses are usually auth walls

# AEM-internal URL pattern. Capital Group's AEM author paths look like
# /content/capital-group/us/en/<segment>/home/insights/articles/<rest>
# and the user-facing equivalent is /<segment>/insights/articles/<rest>.
AEM_PATTERN = re.compile(
    r"^https?://www\.capitalgroup\.com"
    r"/content/capital-group/us/en/([^/]+)/home/insights/articles/(.+)$"
)

# Strings that indicate the Wayback snapshot itself failed or that the
# extracted text is Wayback chrome rather than article body.
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
    """If `url` is the AEM-internal form, return the user-facing equivalent.

    Returns None if the URL doesn't match the AEM pattern.
    """
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


def cdx_query(url: str) -> list[dict]:
    """Query the Wayback CDX API for 200-status snapshots of `url`.

    Returns a list of dicts with keys: timestamp, original, length.
    Returns [] on any failure or if no rows match.
    """
    params = {
        "url": url,
        "output": "json",
        "limit": "50",
        "filter": ["statuscode:200", "!mimetype:warc/revisit"],
    }
    try:
        resp = http_get(
            _build_cdx_url(params),
            timeout=CDX_TIMEOUT_SECONDS,
        )
    except (requests.Timeout, requests.ConnectionError, requests.RequestException):
        return []
    if resp.status_code != 200:
        return []
    try:
        rows = resp.json()
    except ValueError:
        return []
    if not isinstance(rows, list) or len(rows) < 2:
        return []
    header, *data_rows = rows
    # Header is typically ["urlkey","timestamp","original","mimetype","statuscode","digest","length"]
    try:
        ts_idx = header.index("timestamp")
        orig_idx = header.index("original")
        status_idx = header.index("statuscode")
        len_idx = header.index("length")
    except ValueError:
        return []
    out: list[dict] = []
    for row in data_rows:
        if len(row) <= max(ts_idx, orig_idx, status_idx, len_idx):
            continue
        if row[status_idx] != "200":
            continue
        try:
            length = int(row[len_idx])
        except (ValueError, TypeError):
            continue
        out.append({
            "timestamp": row[ts_idx],
            "original": row[orig_idx],
            "length": length,
        })
    return out


def _build_cdx_url(params: dict) -> str:
    """Build a CDX URL preserving repeated filter params."""
    parts: list[str] = []
    for key, value in params.items():
        values = value if isinstance(value, list) else [value]
        for v in values:
            parts.append(f"{key}={requests.utils.quote(str(v), safe='')}")
    return f"{CDX_API}?{'&'.join(parts)}"


def pick_snapshot(rows: list[dict]) -> dict | None:
    """Pick the earliest snapshot with length > CDX_MIN_LENGTH.

    Returns None if no row qualifies.
    """
    usable = [r for r in rows if r["length"] > CDX_MIN_LENGTH]
    if not usable:
        return None
    return min(usable, key=lambda r: r["timestamp"])


def wayback_snapshot_url(timestamp: str, original: str) -> str:
    return f"https://web.archive.org/web/{timestamp}/{original}"


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


def body_looks_real(body: str | None) -> bool:
    """True if body is long enough and doesn't look like Wayback chrome."""
    if not body or len(body) < MIN_BODY_CHARS:
        return False
    for marker in WAYBACK_FAILURE_MARKERS:
        if marker in body:
            return False
    return True


def try_source(url: str) -> tuple[int | None, str | None, str | None]:
    """Fetch one URL and extract (http_status, body, title)."""
    http_status, html = try_fetch(url)
    if http_status == 200 and html:
        body, title = extract_body_and_title(html)
        return http_status, body, title
    return http_status, None, None


def fetch_one(post_id: int, url: str, tracks: dict[str, str]) -> dict:
    track = extract_track(url, tracks)
    stripped = strip_query(url)

    # Track URL is what we picked-via-track lookup on (uses the original URL
    # so capitalgroup's /advisor/ /institutional/ /content/ segments all map).
    # Canonical URL only set if we had to canonicalize to find snapshots.

    # Step 1: CDX on the original (query-stripped) URL.
    cdx_rows = cdx_query(stripped)
    time.sleep(INTER_REQUEST_SLEEP_SECONDS)
    snapshot = pick_snapshot(cdx_rows)
    canonical_url: str | None = None

    # Step 2: If nothing usable and URL is AEM-internal, canonicalize and retry.
    if snapshot is None:
        canonical_candidate = canonicalize_aem_url(stripped)
        if canonical_candidate is not None:
            canonical_url = canonical_candidate
            cdx_rows = cdx_query(canonical_candidate)
            time.sleep(INTER_REQUEST_SLEEP_SECONDS)
            snapshot = pick_snapshot(cdx_rows)

    # Step 3: Fetch the chosen snapshot.
    wb_status: int | None = None
    if snapshot is not None:
        snap_url = wayback_snapshot_url(snapshot["timestamp"], snapshot["original"])
        wb_status, wb_body, wb_title = try_source(snap_url)
        if wb_status == 200 and body_looks_real(wb_body):
            return _record(
                post_id=post_id,
                url=url,
                canonical_url=canonical_url,
                source="wayback",
                fetch_status="ok",
                http_status=wb_status,
                body=wb_body or "",
                title=wb_title or slug_to_title(url),
                track=track,
                wayback_timestamp=snapshot["timestamp"],
            )
        time.sleep(INTER_REQUEST_SLEEP_SECONDS)

    # Step 4: Live URL fallback. Use canonical if we derived one, else original.
    live_target = canonical_url or url
    live_status, live_body, live_title = try_source(live_target)
    if live_status == 200 and body_looks_real(live_body):
        return _record(
            post_id=post_id,
            url=url,
            canonical_url=canonical_url,
            source="live",
            fetch_status="ok",
            http_status=live_status,
            body=live_body or "",
            title=live_title or slug_to_title(url),
            track=track,
            wayback_timestamp=None,
        )

    # Step 5: slug-title fallback.
    return _record(
        post_id=post_id,
        url=url,
        canonical_url=canonical_url,
        source="fallback",
        fetch_status="fallback",
        http_status=wb_status if wb_status is not None else live_status,
        body="",
        title=slug_to_title(url),
        track=track,
        wayback_timestamp=None,
    )


def _record(
    *,
    post_id: int,
    url: str,
    canonical_url: str | None,
    source: str,
    fetch_status: str,
    http_status: int | None,
    body: str,
    title: str,
    track: str,
    wayback_timestamp: str | None,
) -> dict:
    record: dict = {
        "postId": int(post_id),
        "url": url,
    }
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
    if source == "wayback" and wayback_timestamp is not None:
        record["wayback_timestamp"] = wayback_timestamp
    return record


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
                f"postId={post_id} source={record['source']} status=ok "
                f"http={record['http_status']} chars={record['char_count']} (short body)"
            )
        else:
            print(
                f"postId={post_id} source={record['source']} "
                f"status={record['fetch_status']} http={record['http_status']} "
                f"chars={record['char_count']}"
            )

        summary = {
            "postId": record["postId"],
            "source": record["source"],
            "fetch_status": record["fetch_status"],
            "char_count": record["char_count"],
            "audience_track": record["audience_track"],
        }
        if "wayback_timestamp" in record:
            summary["wayback_timestamp"] = record["wayback_timestamp"]
        if "canonical_url" in record:
            summary["canonical_url"] = record["canonical_url"]
        summaries.append(summary)

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
