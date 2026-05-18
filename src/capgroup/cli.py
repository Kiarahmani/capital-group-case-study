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
    model: Optional[str] = typer.Option(None, "--model", help="Override the model in config."),
    max_article_chars: Optional[int] = typer.Option(None, "--max-article-chars", help="Truncate article body. Default: no truncation."),
    retrieval_k: int = typer.Option(8, "--retrieval-k", help="Number of train posts to retrieve as examples."),
    temperature: float = typer.Option(0.7, "--temperature"),
    output_prefix: Optional[Path] = typer.Option(None, "--output-prefix", help="Output prefix. Default: outputs/gen_{timestamp}"),
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
    )


def main():
    app()


if __name__ == "__main__":
    main()
