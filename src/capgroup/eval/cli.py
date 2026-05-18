"""Typer CLI for the held-out evaluation pipeline."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from capgroup.eval.holdout import REPO_ROOT, TRAIN_XLSX, run_holdout_eval, sample_holdout_urls
from capgroup.gen import load_config

app = typer.Typer(add_completion=False)


@app.command()
def evaluate(
    n_urls_advisor: int = typer.Option(7, "--n-advisor"),
    n_urls_institutional: int = typer.Option(2, "--n-institutional"),
    n_urls_content: int = typer.Option(1, "--n-content"),
    min_paraphrases: int = typer.Option(3, "--min-paraphrases", help="Bias sampling toward URLs with at least this many real posts."),
    seed: int = typer.Option(2026, "--seed"),
    variants: int = typer.Option(4, "--variants"),
    flavors: Optional[str] = typer.Option("all", "--flavors", help="Same syntax as capgroup-gen; default 'all'."),
    gen_model: Optional[str] = typer.Option(None, "--gen-model"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model"),
    output_prefix: Optional[Path] = typer.Option(None, "--output-prefix"),
):
    config = load_config()
    tracks = config["audience_tracks"]
    per_track = {
        "advisor": n_urls_advisor,
        "institutional": n_urls_institutional,
        "content": n_urls_content,
    }

    train_df = pd.read_excel(TRAIN_XLSX, sheet_name="nearShoreTrain")
    holdout_urls = sample_holdout_urls(train_df, tracks, per_track, min_paraphrases, seed)
    flavor_names = [f.strip() for f in flavors.split(",")] if flavors else None

    if output_prefix is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_prefix = REPO_ROOT / "outputs" / f"eval_{ts}"

    print(f"Held-out URLs ({len(holdout_urls)}):")
    for u in holdout_urls:
        track = next((t for t, p in tracks.items() if p in u), "other")
        n = (train_df["URL"] == u).sum()
        print(f"  [{track}] {u}  ({n} real posts)")
    print()

    run_holdout_eval(
        holdout_urls=holdout_urls,
        n_variants=variants,
        flavor_names=flavor_names,
        output_prefix=output_prefix,
        gen_model_override=gen_model,
        judge_model_override=judge_model,
    )


def main():
    app()


if __name__ == "__main__":
    main()
