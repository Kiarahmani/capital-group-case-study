# Capital Group — Social Media Post Generation

A retrieval-augmented pipeline that drafts short social media posts (LinkedIn / X) promoting Capital Group articles. Given a target article URL and a historical corpus of human-written posts, it produces N candidate posts per article, each scored and gated through an LLM-as-judge quality check before being written out.

## How to run

```bash
# 1. Install
uv sync

# 2. Set keys
cp .env.example .env  # fill in ANTHROPIC_API_KEY and VOYAGE_API_KEY

# 3. Drop the input files into data/inputs/
#    (train.xlsx and test.xlsx are expected but not committed)

# 4. Generate posts for all test articles, 4 variants each, with flavor tags
uv run capgroup-gen --flavors all
# Output: outputs/gen_{timestamp}_posts.csv + .xlsx + _trace.jsonl

# 5. Held-out evaluation
uv run capgroup-eval
# Output: outputs/eval_{timestamp}.csv + .md
```

### Pre-generation steps (optional — outputs are committed)

The repo ships with pre-generation outputs already in place (`data/cache/articles/*.json`, `data/cache/embeddings.npz`, plus the audit files under `outputs/`). Rebuild only if you've deleted them or want to regenerate from scratch:

```bash
uv run python -m capgroup.pregen.style_audit
uv run python -m capgroup.pregen.disclosure_urls     # skip-if-populated; --force to overwrite
uv run python -m capgroup.pregen.fetch_articles      # skip-if-exists; --force to rebuild
uv run python -m capgroup.pregen.embed
```

Notes:
- `fetch_articles.py` — Wayback Machine is non-deterministic; the same script can return different snapshots at different times. The committed article cache pins the specific snapshots used in the canonical run (each file records its `source: "wayback:<timestamp>"`). The script skips postIds whose cache already exists; pass `--force` to overwrite.
- `disclosure_urls.py` — frequency-based URL extraction can surface near-duplicate truncation artifacts in dirty source data. The committed `config.yaml` has been manually deduped to the single canonical URL. The script skips if `disclosure_links` is already populated; pass `--force` to re-extract.

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

## Configuration

All tunables in [`config.yaml`](config.yaml) — model defaults, retrieval parameters, flavor tag descriptions, judge calibration examples, canonical disclosure URL.

## Sample outputs

The repo ships with the canonical outputs from one full run:

- [`outputs/gen_20260519T180346Z_trace.jsonl`](outputs/gen_20260519T180346Z_trace.jsonl) — per-call provenance for the canonical run: retrieved example postIds, retry attempts, judge notes, token usage.
- [`outputs/eval_20260519T172959Z.md`](outputs/eval_20260519T172959Z.md) + [`.csv`](outputs/eval_20260519T172959Z.csv) — held-out evaluation summary and per-post metrics.
- [`outputs/style_audit.md`](outputs/style_audit.md) + `style_audit.json` — deterministic corpus statistics.
- [`outputs/disclosure_url_audit.md`](outputs/disclosure_url_audit.md) — canonical disclosure URL extraction.

## Design notes

See [`outputs/one_pager.txt`](outputs/one_pager.txt) for the approach, findings, and stack discussion.

## Stack

Python 3.11 · `uv` · Claude Sonnet 4.6 (Anthropic SDK with prompt caching and tool use) · Voyage 3.5 (asymmetric document/query embeddings) · trafilatura · pandas · typer · openpyxl.
