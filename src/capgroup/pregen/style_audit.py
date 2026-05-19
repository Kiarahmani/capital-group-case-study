"""Deterministic style audit of the train corpus — pandas + regex + json, no LLM."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config.yaml"
TRAIN_PATH = REPO_ROOT / "data" / "inputs" / "train.xlsx"
TRAIN_SHEET = "nearShoreTrain"
OUT_JSON = REPO_ROOT / "outputs" / "style_audit.json"
OUT_MD = REPO_ROOT / "outputs" / "style_audit.md"

HASHTAG_RE = re.compile(r"#\w+")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
CAPITALGROUP_RE = re.compile(r"https?://[^\s]*capitalgroup\.com", re.IGNORECASE)
BITLY_RE = re.compile(r"https?://bit\.ly/\S+", re.IGNORECASE)
DISCLOSURE_RE = re.compile(r"important disclosures", re.IGNORECASE)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")
QUESTION_RE = re.compile(r"\?")
ALLCAPS_RE = re.compile(r"\b[A-Z]{3,}\b")


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def classify_track(url: str, tracks: dict[str, str]) -> str:
    if not isinstance(url, str):
        return "other"
    for name, fragment in tracks.items():
        if fragment in url:
            return name
    return "other"


def strip_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def fraction(mask: pd.Series) -> float:
    if len(mask) == 0:
        return 0.0
    return float(mask.mean())


def compute_metrics(posts: pd.Series) -> dict:
    n = len(posts)
    if n == 0:
        return {"count": 0}
    lengths = posts.str.len()
    hashtag_counts = posts.apply(lambda t: len(HASHTAG_RE.findall(t)))
    has_hashtag = hashtag_counts > 0
    has_capitalgroup = posts.apply(lambda t: bool(CAPITALGROUP_RE.search(t)))
    has_bitly = posts.apply(lambda t: bool(BITLY_RE.search(t)))
    has_any_url = posts.apply(lambda t: bool(URL_RE.search(t)))
    has_disclosure = posts.apply(lambda t: bool(DISCLOSURE_RE.search(t)))
    has_emoji = posts.apply(lambda t: bool(EMOJI_RE.search(t)))
    has_question = posts.apply(lambda t: bool(QUESTION_RE.search(t)))
    has_allcaps = posts.apply(lambda t: bool(ALLCAPS_RE.search(strip_urls(t))))

    return {
        "count": int(n),
        "length": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "median": float(lengths.median()),
            "p25": float(lengths.quantile(0.25)),
            "p75": float(lengths.quantile(0.75)),
        },
        "hashtag": {
            "presence_rate": fraction(has_hashtag),
            "avg_count_per_post": float(hashtag_counts.mean()),
        },
        "url": {
            "full_capitalgroup_rate": fraction(has_capitalgroup),
            "bitly_rate": fraction(has_bitly),
            "any_url_rate": fraction(has_any_url),
            "no_url_rate": 1.0 - fraction(has_any_url),
        },
        "disclosure_trailer_rate": fraction(has_disclosure),
        "emoji_rate": fraction(has_emoji),
        "question_mark_rate": fraction(has_question),
        "allcaps_word_rate": fraction(has_allcaps),
    }


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def format_num(x: float) -> str:
    if isinstance(x, int) or (isinstance(x, float) and x.is_integer()):
        return f"{int(x)}"
    return f"{x:.2f}"


def render_slice_md(name: str, m: dict) -> str:
    if m.get("count", 0) == 0:
        return f"## {name}\n\n(no rows)\n"
    lines = [
        f"## {name}",
        "",
        f"- count: {m['count']}",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    L = m["length"]
    lines.append(
        f"| length min / median / max | {format_num(L['min'])} / {format_num(L['median'])} / {format_num(L['max'])} |"
    )
    lines.append(
        f"| length p25 / p75 | {format_num(L['p25'])} / {format_num(L['p75'])} |"
    )
    lines.append(
        f"| hashtag presence rate | {format_pct(m['hashtag']['presence_rate'])} |"
    )
    lines.append(
        f"| hashtag avg count per post | {m['hashtag']['avg_count_per_post']:.2f} |"
    )
    lines.append(
        f"| url full capitalgroup.com rate | {format_pct(m['url']['full_capitalgroup_rate'])} |"
    )
    lines.append(f"| url bit.ly rate | {format_pct(m['url']['bitly_rate'])} |")
    lines.append(f"| url any-url rate | {format_pct(m['url']['any_url_rate'])} |")
    lines.append(f"| url no-url rate | {format_pct(m['url']['no_url_rate'])} |")
    lines.append(
        f"| disclosure trailer rate | {format_pct(m['disclosure_trailer_rate'])} |"
    )
    lines.append(f"| emoji rate | {format_pct(m['emoji_rate'])} |")
    lines.append(f"| question mark rate | {format_pct(m['question_mark_rate'])} |")
    lines.append(f"| allcaps word rate | {format_pct(m['allcaps_word_rate'])} |")
    lines.append("")
    return "\n".join(lines)


def cross_track_differences(slices: dict) -> list[str]:
    track_names = [
        t
        for t in ("advisor", "institutional", "content")
        if t in slices and slices[t].get("count", 0) > 0
    ]
    if len(track_names) < 2:
        return []

    notes: list[str] = []

    def spread(getter) -> tuple[float, str, float, str, float]:
        values = [(name, getter(slices[name])) for name in track_names]
        lo_name, lo = min(values, key=lambda v: v[1])
        hi_name, hi = max(values, key=lambda v: v[1])
        return hi - lo, lo_name, lo, hi_name, hi

    rate_metrics = [
        ("hashtag presence rate", lambda s: s["hashtag"]["presence_rate"]),
        ("hashtag avg count per post", lambda s: s["hashtag"]["avg_count_per_post"]),
        (
            "full capitalgroup.com URL rate",
            lambda s: s["url"]["full_capitalgroup_rate"],
        ),
        ("bit.ly URL rate", lambda s: s["url"]["bitly_rate"]),
        ("any-URL rate", lambda s: s["url"]["any_url_rate"]),
        ("disclosure trailer rate", lambda s: s["disclosure_trailer_rate"]),
        ("emoji rate", lambda s: s["emoji_rate"]),
        ("question mark rate", lambda s: s["question_mark_rate"]),
        ("allcaps word rate", lambda s: s["allcaps_word_rate"]),
    ]
    for label, getter in rate_metrics:
        gap, lo_name, lo, hi_name, hi = spread(getter)
        if label == "hashtag avg count per post":
            if gap > 0.15:
                notes.append(
                    f"- {label}: {hi_name} {hi:.2f} vs {lo_name} {lo:.2f} (spread {gap:.2f})."
                )
        elif gap > 0.15:
            notes.append(
                f"- {label}: {hi_name} {format_pct(hi)} vs {lo_name} {format_pct(lo)} (spread {format_pct(gap)})."
            )

    for label, getter in [
        ("median length", lambda s: s["length"]["median"]),
        ("p75 length", lambda s: s["length"]["p75"]),
    ]:
        gap, lo_name, lo, hi_name, hi = spread(getter)
        rel = gap / lo if lo > 0 else 0.0
        if rel > 0.25:
            notes.append(
                f"- {label}: {hi_name} {hi:.0f} chars vs {lo_name} {lo:.0f} chars "
                f"(spread {gap:.0f} chars, {rel * 100:.0f}% over the lower)."
            )

    return notes


def render_markdown(slices: dict, meta: dict) -> str:
    parts = [
        "# Style audit — train corpus",
        "",
        f"- generated_at: {meta['generated_at']}",
        f"- train_path: {meta['train_path']}",
        f"- train_rows: {meta['train_rows']}",
        "",
    ]
    order = ["overall", "advisor", "institutional", "content", "other"]
    for name in order:
        if name not in slices:
            continue
        parts.append(render_slice_md(name, slices[name]))

    diffs = cross_track_differences(slices)
    parts.append("## Notable differences across tracks")
    parts.append("")
    if diffs:
        parts.append(
            "Thresholds: rate metrics with >15 percentage-point spread, length metrics with >25% spread over the lower value."
        )
        parts.append("")
        parts.extend(diffs)
    else:
        parts.append(
            "No metric crossed the thresholds (>15pp for rates, >25% for length)."
        )
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    config = load_config(CONFIG_PATH)
    tracks: dict[str, str] = config["audience_tracks"]

    df = pd.read_excel(TRAIN_PATH, sheet_name=TRAIN_SHEET)
    df = df[["postId", "postOriginal", "URL"]].copy()
    df["postOriginal"] = df["postOriginal"].fillna("").astype(str)
    df["URL"] = df["URL"].fillna("").astype(str)
    df["track"] = df["URL"].apply(lambda u: classify_track(u, tracks))

    slices = {"overall": compute_metrics(df["postOriginal"])}
    for track_name in list(tracks.keys()) + ["other"]:
        subset = df[df["track"] == track_name]["postOriginal"]
        if len(subset) == 0 and track_name == "other":
            continue
        slices[track_name] = compute_metrics(subset)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_path": str(TRAIN_PATH.relative_to(REPO_ROOT)),
        "train_rows": int(len(df)),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, **slices}
    with OUT_JSON.open("w") as f:
        json.dump(payload, f, indent=2)

    md = render_markdown(slices, meta)
    with OUT_MD.open("w") as f:
        f.write(md)

    overall = slices["overall"]
    print(
        f"overall: {overall['count']} rows, "
        f"median length {overall['length']['median']:.0f} chars, "
        f"hashtag rate {overall['hashtag']['presence_rate'] * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
