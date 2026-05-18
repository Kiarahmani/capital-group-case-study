"""Typer CLI for the generation pipeline."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from capgroup.gen import REPO_ROOT, list_test_post_ids, run_generation

app = typer.Typer(add_completion=False)


@app.command()
def generate(
    post_ids: Optional[str] = typer.Option(None, "--post-ids", help="Comma-separated test postIds. Default: all 16."),
    variants: int = typer.Option(4, "--variants", help="Variants per article."),
    flavors: Optional[str] = typer.Option(None, "--flavors", help="Comma-separated flavor tag names from config, or 'all'."),
    include_disclosure: bool = typer.Option(False, "--include-disclosure", help="Append the canonical disclosure trailer to each post."),
    model: Optional[str] = typer.Option(None, "--model", help="Override the gen model in config."),
    max_article_chars: Optional[int] = typer.Option(None, "--max-article-chars", help="Truncate article body. Default: no truncation."),
    retrieval_k: int = typer.Option(8, "--retrieval-k", help="Number of train posts to retrieve as examples."),
    temperature: float = typer.Option(0.7, "--temperature"),
    output_prefix: Optional[Path] = typer.Option(None, "--output-prefix", help="Output prefix. Default: outputs/gen_{timestamp}"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip the judge + retry loop. Faster, no quality gating."),
    max_judge_retries: Optional[int] = typer.Option(None, "--max-judge-retries", help="Override judge retry cap from config (default 2)."),
    length_min: Optional[int] = typer.Option(None, "--length-min", help="Hard min char length. Override config."),
    length_max: Optional[int] = typer.Option(None, "--length-max", help="Hard max char length. Override config."),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="Override the judge model in config."),
):
    ids = [int(x.strip()) for x in post_ids.split(",")] if post_ids else list_test_post_ids()
    flavor_names = [f.strip() for f in flavors.split(",")] if flavors else None

    if output_prefix is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_prefix = REPO_ROOT / "outputs" / f"gen_{ts}"

    run_generation(
        post_ids=ids,
        n_variants=variants,
        flavor_names=flavor_names,
        include_disclosure=include_disclosure,
        model=model,
        max_article_chars=max_article_chars,
        retrieval_k=retrieval_k,
        temperature=temperature,
        output_prefix=output_prefix,
        use_judge=not no_judge,
        max_judge_retries=max_judge_retries,
        length_min=length_min,
        length_max=length_max,
        judge_model=judge_model,
    )


def main():
    app()


if __name__ == "__main__":
    main()
