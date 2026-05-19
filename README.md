# Capital Group — Social Media Post Generation Case Study

Pipeline that generates short social media posts (LinkedIn / X) promoting Capital Group articles, given an article URL. Built for the Capital Group AI Scientist case study, May 2026.

## Writeup

See **[`outputs/one_pager.txt`](outputs/one_pager.txt)** for the full approach, findings, limitations, and what-I'd-do-next.

## How to run

```bash
# 1. Install
uv sync

# 2. Set keys
cp .env.example .env  # then fill in ANTHROPIC_API_KEY and VOYAGE_API_KEY

# 3. Drop the case-study input files into data/inputs/
#    (train.xlsx, test.xlsx — these are not committed to the repo)

# 4. Generate posts for all 16 test articles, 4 variants each, with flavor tags
uv run capgroup-gen --flavors all
# Output: outputs/gen_{timestamp}_posts.csv + .xlsx + _trace.jsonl

# 5. Held-out evaluation
uv run capgroup-eval
# Output: outputs/eval_{timestamp}.csv + .md
```

### Pre-generation steps (optional — outputs are committed)

The repo ships with the pre-generation outputs already committed
(`data/cache/articles/*.json`, `data/cache/embeddings.npz`, plus the
audit files under `outputs/`). You only need to rebuild these if you've
deleted them or want to regenerate from scratch:

```bash
uv run python -m capgroup.pregen.style_audit
uv run python -m capgroup.pregen.disclosure_urls     # skip-if-populated; --force to overwrite
uv run python -m capgroup.pregen.fetch_articles      # skip-if-exists; --force to rebuild
uv run python -m capgroup.pregen.embed
```

Notes:
- `fetch_articles.py` — Wayback Machine is non-deterministic; the same
  script can return different snapshots at different times. The committed
  article cache pins the specific snapshots used in the submission run
  (each file records its `source: "wayback:<timestamp>"`). The script
  skips postIds whose cache already exists; pass `--force` to overwrite.
- `disclosure_urls.py` — frequency-based URL extraction can surface
  near-duplicate truncation artifacts in the source data. The committed
  `config.yaml` has been manually deduped to the single canonical URL.
  The script skips if `disclosure_links` is already populated; pass
  `--force` to re-extract.

## Engineering evidence (committed to repo)

- **[`outputs/one_pager.txt`](outputs/one_pager.txt)** — the writeup.
- **[`outputs/eval_20260518T232813Z.md`](outputs/eval_20260518T232813Z.md)** — held-out evaluation summary (gen vs human cluster on cosine + judge rubric).
- **[`outputs/eval_20260518T232813Z.csv`](outputs/eval_20260518T232813Z.csv)** — per-generated-post metrics from the eval run.
- **[`outputs/gen_20260518T232812Z_trace.jsonl`](outputs/gen_20260518T232812Z_trace.jsonl)** — per-call provenance for the final Sonnet run: retrieved example postIds, retry attempts, judge notes, token usage.
- **[`outputs/style_audit.md`](outputs/style_audit.md)** + `style_audit.json` — deterministic corpus statistics.
- **[`outputs/disclosure_url_audit.md`](outputs/disclosure_url_audit.md)** — canonical disclosure URL extraction.

## Deliverable (ships via email, not the repo)

- `outputs/gen_{timestamp}_posts.csv` / `.xlsx` — the long-format spreadsheet (postId, variantId, flavor, audienceTrack, articleUrl, generatedPost, plus judge columns). Sent as an email attachment.
- `outputs/one_pager.pdf` — formatted writeup. Email attachment.

## Architecture

```
src/capgroup/
├── pregen/
│   ├── style_audit.py        # deterministic corpus statistics
│   ├── disclosure_urls.py    # extract canonical disclosure URL
│   ├── fetch_articles.py     # Wayback-based article body acquisition
│   └── embed.py              # Voyage 3.5 embeddings for train + test
├── retrieval.py              # stratified kNN with audience-track filter + dedupe-by-URL
├── gen.py                    # generation + inline quality gate + retry loop
├── judge.py                  # LLM-as-judge via Claude tool-use (forced structured output)
├── cli.py                    # capgroup-gen entry point
└── eval/
    ├── holdout.py            # held-out URL eval with cluster-based similarity
    └── cli.py                # capgroup-eval entry point
```

## Config

All tunables in [`config.yaml`](config.yaml) — model defaults, retrieval parameters, flavor tag descriptions, judge calibration examples, canonical disclosure URL.

## Stack

Python 3.11 · `uv` · Claude Sonnet 4.6 (Anthropic SDK with prompt caching and tool use) · Voyage 3.5 (asymmetric document/query embeddings) · trafilatura · pandas · typer.
