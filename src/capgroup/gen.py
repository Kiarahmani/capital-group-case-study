"""Generation pipeline: prompt build, Claude call with caching, parse."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
import pandas as pd
import yaml
from dotenv import load_dotenv

from capgroup.retrieval import Retriever

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"
ARTICLES_DIR = REPO_ROOT / "data" / "cache" / "articles"
ENV_PATH = REPO_ROOT / ".env"

SYSTEM_PROMPT_TEMPLATE = """You are a marketing copywriter for Capital Group, a regulated US asset manager.

Your job is to draft short social media posts (LinkedIn or X) that promote a Capital Group article. The post will be reviewed and edited by a human marketing team member before publishing — your output is a starting draft, not finished publish-ready text.

Hard constraints:
- No emojis.
- No specific investment return claims (no "guaranteed X%", no "will return Y%").
- No buy/sell recommendations, no specific stock tickers.
- Output only the post text. No preamble, no quotes wrapping the post, no commentary.

Voice and style:
- Match the style of the provided examples, which are real human-written posts from Capital Group's social media accounts.
- Length: {length_hint}.
- Hashtag use, question hooks, URL placement, and other stylistic choices should follow what the examples show.
"""

DISCLOSURE_INSTRUCTION = 'Append exactly this line to the end of your post, on a new line: "Important disclosures: {url}"'


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def load_article(post_id: int) -> dict:
    return json.loads((ARTICLES_DIR / f"{post_id}.json").read_text())


def list_test_post_ids() -> list[int]:
    idx = json.loads((ARTICLES_DIR / "_index.json").read_text())
    return sorted(a["postId"] for a in idx["articles"])


def build_user_blocks(
    article: dict,
    examples: list[dict],
    max_article_chars: Optional[int],
    flavor: Optional[dict],
    include_disclosure: bool,
    disclosure_url: Optional[str],
) -> list[dict]:
    """Build user content blocks. Cache breakpoint on the large reusable section."""
    body = article["body"] or article.get("title", "")
    if max_article_chars and len(body) > max_article_chars:
        body = body[:max_article_chars] + "..."

    examples_section = "\n\n".join(
        f"<example>\n{ex['text_clean']}\n</example>" for ex in examples
    )

    cached_text = f"""Audience track: {article['audience_track']}

Examples of social media posts from this audience track:

{examples_section}

The article you are promoting:
Title: {article.get('title', '')}
URL: {article['url']}

Body:
{body}"""

    instructions = ["Write one social media post for this article."]
    if flavor:
        instructions.append(f"Style direction: {flavor['description']}")
    if include_disclosure and disclosure_url:
        instructions.append(DISCLOSURE_INSTRUCTION.format(url=disclosure_url))
    delta_text = "\n\n".join(instructions)

    return [
        {"type": "text", "text": cached_text, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": delta_text},
    ]


def parse_response(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.startswith("```")]
        text = "\n".join(lines).strip()
    return text


def generate_one(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    user_blocks: list[dict],
    temperature: float,
    max_tokens: int,
) -> dict:
    t0 = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_blocks}],
    )
    latency_ms = int((time.time() - t0) * 1000)
    raw_text = response.content[0].text
    return {
        "generated_post": parse_response(raw_text),
        "raw_response": raw_text,
        "latency_ms": latency_ms,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
        },
    }


def select_flavors(n_variants: int, flavors_pool: Optional[list[dict]]) -> list[Optional[dict]]:
    if not flavors_pool:
        return [None] * n_variants
    return [flavors_pool[i % len(flavors_pool)] for i in range(n_variants)]


def resolve_flavor_pool(flavor_names: Optional[list[str]], config: dict) -> Optional[list[dict]]:
    if not flavor_names:
        return None
    if flavor_names == ["all"]:
        return [{"name": k, "description": v} for k, v in config["flavors"].items()]
    return [{"name": k, "description": config["flavors"][k]} for k in flavor_names]


def run_generation(
    post_ids: list[int],
    n_variants: int,
    flavor_names: Optional[list[str]],
    include_disclosure: bool,
    model: Optional[str],
    max_article_chars: Optional[int],
    retrieval_k: int,
    temperature: float,
    output_prefix: Path,
) -> tuple[list[dict], list[dict]]:
    load_dotenv(ENV_PATH)
    config = load_config()

    model = model or config["models"]["generation"]
    length_hint = config["generation"]["length_hint"]
    max_tokens = config["generation"]["max_tokens"]
    disclosure_url = config["disclosure_links"][0] if config.get("disclosure_links") else None

    if include_disclosure and not disclosure_url:
        raise ValueError("include_disclosure set but config.yaml has no disclosure_links")

    flavors_pool = resolve_flavor_pool(flavor_names, config)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(length_hint=length_hint)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    retriever = Retriever()

    rows: list[dict] = []
    trace: list[dict] = []

    for post_id in post_ids:
        article = load_article(post_id)
        examples = retriever.retrieve(post_id, k=retrieval_k, same_track_only=True, dedupe_by_url=True)
        flavors_for_variants = select_flavors(n_variants, flavors_pool)

        top_sim = f"{examples[0]['similarity']:.3f}" if examples else "n/a"
        print(f"\n=== postId={post_id} track={article['audience_track']} ===")
        print(f"    retrieved {len(examples)} examples (top sim: {top_sim})")

        for variant_idx, flavor in enumerate(flavors_for_variants, start=1):
            user_blocks = build_user_blocks(
                article=article,
                examples=examples,
                max_article_chars=max_article_chars,
                flavor=flavor,
                include_disclosure=include_disclosure,
                disclosure_url=disclosure_url,
            )
            result = generate_one(client, model, system_prompt, user_blocks, temperature, max_tokens)

            flavor_name = flavor["name"] if flavor else ""
            row = {
                "postId": post_id,
                "variantId": variant_idx,
                "flavor": flavor_name,
                "audienceTrack": article["audience_track"],
                "articleUrl": article["url"],
                "generatedPost": result["generated_post"],
            }
            rows.append(row)

            u = result["usage"]
            if u["cache_read_input_tokens"]:
                cache_tag = f"READ({u['cache_read_input_tokens']})"
            elif u["cache_creation_input_tokens"]:
                cache_tag = f"WRITE({u['cache_creation_input_tokens']})"
            else:
                cache_tag = "MISS"
            print(
                f"    v{variant_idx} flavor={flavor_name or '-':<14} "
                f"chars={len(result['generated_post']):>4} {cache_tag:<14} "
                f"in/out={u['input_tokens']}/{u['output_tokens']} ({result['latency_ms']}ms)"
            )

            trace.append({
                **row,
                "model": model,
                "raw_response": result["raw_response"],
                "retrieved_post_ids": [e["train_post_id"] for e in examples],
                "retrieved_similarities": [e["similarity"] for e in examples],
                "usage": u,
                "latency_ms": result["latency_ms"],
            })

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_name(output_prefix.name + "_posts.csv")
    xlsx_path = output_prefix.with_name(output_prefix.name + "_posts.xlsx")
    trace_path = output_prefix.with_name(output_prefix.name + "_trace.jsonl")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    with trace_path.open("w") as f:
        for entry in trace:
            f.write(json.dumps(entry) + "\n")

    print("\n=== Run complete ===")
    print(f"  posts: {len(rows)} ({len(post_ids)} articles × {n_variants} variants)")
    print(f"  csv:   {csv_path}")
    print(f"  xlsx:  {xlsx_path}")
    print(f"  trace: {trace_path}")

    return rows, trace
