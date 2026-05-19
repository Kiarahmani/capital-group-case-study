"""Generation pipeline: prompt build, Claude call with caching, parse."""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anthropic
import pandas as pd
import yaml
from dotenv import load_dotenv

from capgroup.judge import JudgeResult, LengthCheck, check_length, judge_post
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
    retry_feedback: Optional[str] = None,
) -> list[dict]:
    """Build user content blocks. Cache breakpoint on the large reusable section.

    The retry_feedback parameter, when set, is appended to the uncached delta only —
    keeping the cached chunk identical across retries of the same post.
    """
    body = article["body"] or article.get("title", "")
    if max_article_chars and len(body) > max_article_chars:
        body = body[:max_article_chars] + "..."

    examples_section = "\n\n".join(
        f"<example>\n{ex['text_clean']}\n</example>" for ex in examples
    )

    cached_text = f"""Audience track: {article["audience_track"]}

Examples of social media posts from this audience track:

{examples_section}

The article you are promoting:
Title: {article.get("title", "")}
URL: {article["url"]}

Body:
{body}"""

    instructions = ["Write one social media post for this article."]
    if flavor:
        instructions.append(f"Style direction: {flavor['description']}")
    if include_disclosure and disclosure_url:
        instructions.append(DISCLOSURE_INSTRUCTION.format(url=disclosure_url))
    if retry_feedback:
        instructions.append(
            f"Important: a previous attempt failed quality checks. Issue: {retry_feedback} "
            f"Produce a new post that addresses this issue."
        )
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
        lines = [line for line in text.split("\n") if not line.startswith("```")]
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
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
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
            "cache_creation_input_tokens": getattr(
                response.usage, "cache_creation_input_tokens", 0
            ),
            "cache_read_input_tokens": getattr(
                response.usage, "cache_read_input_tokens", 0
            ),
        },
    }


def select_flavors(
    n_variants: int, flavors_pool: Optional[list[dict]]
) -> list[Optional[dict]]:
    if not flavors_pool:
        return [None] * n_variants
    return [flavors_pool[i % len(flavors_pool)] for i in range(n_variants)]


@dataclass
class QualityGateResult:
    """Bundled outcome of generate → judge → retry."""

    post: str
    raw_response: str
    length: LengthCheck
    judgment: JudgeResult
    attempts: list[dict]
    flagged: bool
    retries_used: int
    gen_usage_total: dict
    judge_usage_total: dict


def _empty_usage() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def _accumulate_usage(acc: dict, delta: dict) -> None:
    for k in acc:
        acc[k] += delta.get(k, 0)


def generate_with_quality_gate(
    client: anthropic.Anthropic,
    gen_model: str,
    judge_model: str,
    system_prompt: str,
    article: dict,
    examples: list[dict],
    calibration_examples: list[dict],
    max_article_chars: Optional[int],
    flavor: Optional[dict],
    include_disclosure: bool,
    disclosure_url: Optional[str],
    temperature: float,
    max_tokens: int,
    length_min: int,
    length_max: int,
    max_retries: int,
) -> QualityGateResult:
    """Gen → length check → judge → retry on hard-gate failure. Returns best attempt."""
    attempts: list[dict] = []
    retry_feedback: Optional[str] = None
    gen_usage = _empty_usage()
    judge_usage = _empty_usage()

    for attempt_idx in range(max_retries + 1):
        user_blocks = build_user_blocks(
            article=article,
            examples=examples,
            max_article_chars=max_article_chars,
            flavor=flavor,
            include_disclosure=include_disclosure,
            disclosure_url=disclosure_url,
            retry_feedback=retry_feedback,
        )
        gen_result = generate_one(
            client, gen_model, system_prompt, user_blocks, temperature, max_tokens
        )
        post = gen_result["generated_post"]
        length_result = check_length(post, length_min, length_max)
        judgment = judge_post(
            client, judge_model, article, examples, calibration_examples, post
        )

        _accumulate_usage(gen_usage, gen_result["usage"])
        _accumulate_usage(judge_usage, judgment.usage)

        attempts.append(
            {
                "attempt": attempt_idx + 1,
                "post": post,
                "length": length_result,
                "judgment": judgment,
                "raw_response": gen_result["raw_response"],
                "retry_feedback_used": retry_feedback,
            }
        )

        if (
            length_result.in_envelope
            and judgment.compliance == "pass"
            and judgment.is_valid_post == "pass"
        ):
            return QualityGateResult(
                post=post,
                raw_response=gen_result["raw_response"],
                length=length_result,
                judgment=judgment,
                attempts=attempts,
                flagged=False,
                retries_used=attempt_idx,
                gen_usage_total=gen_usage,
                judge_usage_total=judge_usage,
            )

        feedback_parts = []
        if not length_result.in_envelope:
            feedback_parts.append(length_result.feedback)
        if judgment.compliance == "fail":
            feedback_parts.append(f"Compliance issue: {judgment.compliance_notes}")
        if judgment.is_valid_post == "fail":
            feedback_parts.append(
                f"Output format issue: your previous attempt was not a publishable post. "
                f"Issue: {judgment.is_valid_post_notes}. Output ONLY the social media post "
                f"itself — no preamble, no operator-directed text, no explanations about "
                f"missing information. If specific facts you'd want to use aren't available, "
                f"use a general topical framing instead."
            )
        retry_feedback = " ".join(feedback_parts)

    def score_attempt(a):
        return (
            int(a["length"].in_envelope),
            int(a["judgment"].compliance == "pass"),
            int(a["judgment"].is_valid_post == "pass"),
            a["judgment"].soft_score_sum,
        )

    best = max(attempts, key=score_attempt)
    return QualityGateResult(
        post=best["post"],
        raw_response=best["raw_response"],
        length=best["length"],
        judgment=best["judgment"],
        attempts=attempts,
        flagged=True,
        retries_used=max_retries,
        gen_usage_total=gen_usage,
        judge_usage_total=judge_usage,
    )


def resolve_flavor_pool(
    flavor_names: Optional[list[str]], config: dict
) -> Optional[list[dict]]:
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
    use_judge: bool = True,
    max_judge_retries: Optional[int] = None,
    length_min: Optional[int] = None,
    length_max: Optional[int] = None,
    judge_model: Optional[str] = None,
) -> tuple[list[dict], list[dict]]:
    load_dotenv(ENV_PATH)
    config = load_config()

    gen_model = model or config["models"]["generation"]
    judge_model = judge_model or config["models"]["judge"]
    length_hint = config["generation"]["length_hint"]
    max_tokens = config["generation"]["max_tokens"]
    disclosure_url = (
        config["disclosure_links"][0] if config.get("disclosure_links") else None
    )

    judge_cfg = config.get("judge", {})
    judge_enabled = use_judge and judge_cfg.get("enabled", True)
    max_retries = (
        max_judge_retries
        if max_judge_retries is not None
        else judge_cfg.get("max_retries", 2)
    )
    length_min_eff = (
        length_min if length_min is not None else judge_cfg.get("length_min", 100)
    )
    length_max_eff = (
        length_max if length_max is not None else judge_cfg.get("length_max", 350)
    )
    calibration_examples = judge_cfg.get("calibration_examples", [])

    if include_disclosure and not disclosure_url:
        raise ValueError(
            "include_disclosure set but config.yaml has no disclosure_links"
        )

    flavors_pool = resolve_flavor_pool(flavor_names, config)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(length_hint=length_hint)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    retriever = Retriever()

    rows: list[dict] = []
    trace: list[dict] = []

    print(
        f"Judge: {'ENABLED' if judge_enabled else 'DISABLED'} "
        f"(max_retries={max_retries}, length={length_min_eff}-{length_max_eff})"
    )

    for post_id in post_ids:
        article = load_article(post_id)
        examples = retriever.retrieve(
            post_id, k=retrieval_k, same_track_only=True, dedupe_by_url=True
        )
        flavors_for_variants = select_flavors(n_variants, flavors_pool)

        top_sim = f"{examples[0]['similarity']:.3f}" if examples else "n/a"
        print(f"\n=== postId={post_id} track={article['audience_track']} ===")
        print(f"    retrieved {len(examples)} examples (top sim: {top_sim})")

        for variant_idx, flavor in enumerate(flavors_for_variants, start=1):
            flavor_name = flavor["name"] if flavor else ""

            if judge_enabled:
                qg = generate_with_quality_gate(
                    client=client,
                    gen_model=gen_model,
                    judge_model=judge_model,
                    system_prompt=system_prompt,
                    article=article,
                    examples=examples,
                    calibration_examples=calibration_examples,
                    max_article_chars=max_article_chars,
                    flavor=flavor,
                    include_disclosure=include_disclosure,
                    disclosure_url=disclosure_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    length_min=length_min_eff,
                    length_max=length_max_eff,
                    max_retries=max_retries,
                )
                post_text = qg.post
                j = qg.judgment
                row = {
                    "postId": post_id,
                    "variantId": variant_idx,
                    "flavor": flavor_name,
                    "audienceTrack": article["audience_track"],
                    "articleUrl": article["url"],
                    "generatedPost": post_text,
                    "voiceMatch": j.voice_match,
                    "onTopic": j.on_topic,
                    "hookStrength": j.hook_strength,
                    "compliance": j.compliance,
                    "complianceNotes": j.compliance_notes,
                    "isValidPost": j.is_valid_post,
                    "isValidPostNotes": j.is_valid_post_notes,
                    "lengthChars": qg.length.char_count,
                    "judgeRetries": qg.retries_used,
                    "judgeFlagged": qg.flagged,
                }
                rows.append(row)
                print(
                    f"    v{variant_idx} flavor={flavor_name or '-':<14} "
                    f"chars={qg.length.char_count:>4} retries={qg.retries_used} "
                    f"compl={j.compliance} voice/topic/hook={j.voice_match}/{j.on_topic}/{j.hook_strength}"
                    f"{' FLAGGED' if qg.flagged else ''}"
                )
                trace.append(
                    {
                        **row,
                        "model": gen_model,
                        "judgeModel": judge_model,
                        "retrievedPostIds": [e["train_post_id"] for e in examples],
                        "retrievedSimilarities": [e["similarity"] for e in examples],
                        "genUsage": qg.gen_usage_total,
                        "judgeUsage": qg.judge_usage_total,
                        "judgeNotes": j.overall_notes,
                        "attempts": [
                            {
                                "attempt": a["attempt"],
                                "post": a["post"],
                                "length_in_envelope": a["length"].in_envelope,
                                "compliance": a["judgment"].compliance,
                                "compliance_notes": a["judgment"].compliance_notes,
                                "is_valid_post": a["judgment"].is_valid_post,
                                "is_valid_post_notes": a[
                                    "judgment"
                                ].is_valid_post_notes,
                                "voice_match": a["judgment"].voice_match,
                                "on_topic": a["judgment"].on_topic,
                                "hook_strength": a["judgment"].hook_strength,
                                "retry_feedback_used": a["retry_feedback_used"],
                            }
                            for a in qg.attempts
                        ],
                    }
                )
            else:
                user_blocks = build_user_blocks(
                    article=article,
                    examples=examples,
                    max_article_chars=max_article_chars,
                    flavor=flavor,
                    include_disclosure=include_disclosure,
                    disclosure_url=disclosure_url,
                )
                result = generate_one(
                    client,
                    gen_model,
                    system_prompt,
                    user_blocks,
                    temperature,
                    max_tokens,
                )
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
                cache_tag = (
                    f"READ({u['cache_read_input_tokens']})"
                    if u["cache_read_input_tokens"]
                    else f"WRITE({u['cache_creation_input_tokens']})"
                    if u["cache_creation_input_tokens"]
                    else "MISS"
                )
                print(
                    f"    v{variant_idx} flavor={flavor_name or '-':<14} "
                    f"chars={len(result['generated_post']):>4} {cache_tag:<14} "
                    f"in/out={u['input_tokens']}/{u['output_tokens']} ({result['latency_ms']}ms)"
                )
                trace.append(
                    {
                        **row,
                        "model": gen_model,
                        "raw_response": result["raw_response"],
                        "retrievedPostIds": [e["train_post_id"] for e in examples],
                        "retrievedSimilarities": [e["similarity"] for e in examples],
                        "genUsage": u,
                        "latency_ms": result["latency_ms"],
                    }
                )

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
