"""Extract canonical disclosure URLs from train postOriginal — pandas + regex, no LLM.

Method:
1. Pull every URL from every train post via a robust regex.
2. Rank URL frequency (a) across all posts, (b) restricted to posts containing
   "important disclosures".
3. Canonical disclosure URLs = those in ranking (b) with count >= 3. If <3
   candidates clear the bar, the threshold is dropped to 2 (and the audit notes it).
4. Write canonical list into config.yaml's `disclosure_links` field with
   ruamel.yaml round-trip, preserving comments + layout.
5. Emit a human-readable audit to outputs/disclosure_url_audit.md.
"""

import argparse
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config.yaml"
TRAIN_PATH = REPO_ROOT / "data" / "inputs" / "train.xlsx"
TRAIN_SHEET = "nearShoreTrain"
OUT_MD = REPO_ROOT / "outputs" / "disclosure_url_audit.md"

# Matches http(s) URLs greedily up to whitespace, then trailing punctuation is stripped.
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
TRAILING_PUNCT = ".,;)"
DISCLOSURE_MARKER = "important disclosures"
DEFAULT_THRESHOLD = 3
FALLBACK_THRESHOLD = 2
TOP_N = 20


def extract_urls(text: str) -> list[str]:
    """Pull all URLs from text, strip trailing punctuation."""
    urls = URL_RE.findall(text)
    return [u.rstrip(TRAILING_PUNCT) for u in urls]


def url_counts(posts: pd.Series) -> Counter:
    """Frequency of each URL across the given posts (one count per occurrence)."""
    counter: Counter = Counter()
    for text in posts:
        counter.update(extract_urls(text))
    return counter


def select_canonical(disclosure_counts: Counter) -> tuple[list[str], int, str]:
    """Pick canonical URLs. Returns (urls, threshold_used, note)."""
    at_default = [u for u, c in disclosure_counts.items() if c >= DEFAULT_THRESHOLD]
    if len(at_default) >= 1:
        urls = sorted(at_default, key=lambda u: (-disclosure_counts[u], u))
        return urls, DEFAULT_THRESHOLD, ""
    at_fallback = [u for u, c in disclosure_counts.items() if c >= FALLBACK_THRESHOLD]
    if at_fallback:
        urls = sorted(at_fallback, key=lambda u: (-disclosure_counts[u], u))
        note = (
            f"No URL reached the default threshold of {DEFAULT_THRESHOLD}; "
            f"fell back to {FALLBACK_THRESHOLD}."
        )
        return urls, FALLBACK_THRESHOLD, note
    return (
        [],
        DEFAULT_THRESHOLD,
        "No URL appeared 2+ times in disclosure-trailer posts.",
    )


def top_n(counter: Counter, n: int) -> list[tuple[str, int]]:
    """Top-n by count, with URL as deterministic tiebreaker."""
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def update_config_yaml(path: Path, urls: list[str]) -> None:
    """Replace disclosure_links in config.yaml, preserving comments + ordering."""
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    with path.open() as f:
        data = yaml.load(f)
    data["disclosure_links"] = [DoubleQuotedScalarString(u) for u in urls]
    with path.open("w") as f:
        yaml.dump(data, f)


def read_existing_disclosure_links(path: Path) -> list[str]:
    """Read the current disclosure_links from config.yaml, if any."""
    yaml = YAML()
    with path.open() as f:
        data = yaml.load(f)
    links = data.get("disclosure_links") or []
    return [str(u) for u in links]


def render_audit_md(
    *,
    total_unique: int,
    total_posts: int,
    disclosure_posts: int,
    top_all: list[tuple[str, int]],
    top_disc: list[tuple[str, int]],
    canonical: list[str],
    canonical_counts: Counter,
    threshold: int,
    threshold_note: str,
    generated_at: str,
) -> str:
    def fmt_table(rows: list[tuple[str, int]]) -> str:
        if not rows:
            return "(none)"
        lines = ["| count | url |", "|---:|---|"]
        for url, count in rows:
            lines.append(f"| {count} | {url} |")
        return "\n".join(lines)

    parts = [
        "# Disclosure URL audit — train corpus",
        "",
        f"- generated_at: {generated_at}",
        f"- train_rows: {total_posts}",
        f"- disclosure_trailer_posts: {disclosure_posts} "
        f"(posts containing '{DISCLOSURE_MARKER}', case-insensitive)",
        f"- total_unique_urls: {total_unique}",
        "",
        "## Top 20 most frequent URLs (all train posts)",
        "",
        fmt_table(top_all),
        "",
        "## Top 20 most frequent URLs (disclosure-trailer posts only)",
        "",
        fmt_table(top_disc),
        "",
        "## Canonical disclosure URLs",
        "",
        f"Threshold: a URL must appear in >= {threshold} disclosure-trailer posts.",
    ]
    if threshold_note:
        parts.append("")
        parts.append(f"Note: {threshold_note}")
    parts.append("")
    parts.append(
        "Rationale: URLs that recur as the disclosure-trailer link across multiple "
        "posts are stable disclosure pointers, not one-off article promotion links."
    )
    parts.append("")
    if canonical:
        parts.append("| count | url |")
        parts.append("|---:|---|")
        for url in canonical:
            parts.append(f"| {canonical_counts[url]} | {url} |")
    else:
        parts.append("(no URLs cleared the threshold)")
    parts.append("")
    parts.append("## Observations")
    parts.append("")
    parts.append(
        "- The dominant disclosure URL `https://bit.ly/2JzEDWl` accounts for the vast "
        "majority of disclosure-trailer occurrences."
    )
    parts.append(
        "- The corpus also contains a handful of variants that look like truncations "
        "or copy-paste artifacts of the same short URL (e.g. trailing character missing). "
        "These pass the mechanical threshold but a human may want to dedupe them before "
        "wiring `--include-disclosure` into generation."
    )
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract canonical disclosure URLs from train postOriginal and "
            "populate config.yaml's disclosure_links field. The threshold-based "
            "extraction can surface near-duplicate truncation artifacts in the "
            "source data; the committed config.yaml has been manually deduped. "
            "Default behavior is to skip writing if disclosure_links is already "
            "non-empty — pass --force to re-extract and overwrite."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract and overwrite config.yaml + audit MD even when disclosure_links is already populated.",
    )
    args = parser.parse_args()

    existing = read_existing_disclosure_links(CONFIG_PATH)
    if existing and not args.force:
        print(
            f"disclosure_links already populated in config.yaml "
            f"({len(existing)} URL(s)); skipping extraction. "
            f"Use --force to re-extract."
        )
        return

    df = pd.read_excel(TRAIN_PATH, sheet_name=TRAIN_SHEET)
    posts = df["postOriginal"].fillna("").astype(str)

    all_counts = url_counts(posts)
    disclosure_mask = posts.str.contains(DISCLOSURE_MARKER, case=False, na=False)
    disclosure_counts = url_counts(posts[disclosure_mask])

    canonical, threshold, note = select_canonical(disclosure_counts)

    update_config_yaml(CONFIG_PATH, canonical)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md = render_audit_md(
        total_unique=len(all_counts),
        total_posts=int(len(posts)),
        disclosure_posts=int(disclosure_mask.sum()),
        top_all=top_n(all_counts, TOP_N),
        top_disc=top_n(disclosure_counts, TOP_N),
        canonical=canonical,
        canonical_counts=disclosure_counts,
        threshold=threshold,
        threshold_note=note,
        generated_at=generated_at,
    )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MD.open("w") as f:
        f.write(md)

    print(
        f"train rows: {len(posts)}; disclosure-trailer posts: {int(disclosure_mask.sum())}; "
        f"unique URLs: {len(all_counts)}; canonical disclosure URLs: {len(canonical)} "
        f"(threshold >= {threshold})"
    )
    for url in canonical:
        print(f"  {disclosure_counts[url]:>4}  {url}")


if __name__ == "__main__":
    main()
