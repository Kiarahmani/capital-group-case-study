"""Held-out URL evaluation: gen without leakage, compare to real human clusters."""

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
import numpy as np
import pandas as pd
import voyageai
import yaml
from dotenv import load_dotenv

from capgroup.gen import (
    ENV_PATH,
    REPO_ROOT,
    SYSTEM_PROMPT_TEMPLATE,
    generate_with_quality_gate,
    load_config,
    resolve_flavor_pool,
    select_flavors,
)
from capgroup.judge import judge_post
from capgroup.retrieval import Retriever, clean_example

TRAIN_XLSX = REPO_ROOT / "data" / "inputs" / "train.xlsx"


def url_to_title(url: str) -> str:
    """Derive a human-readable title from a Capital Group URL slug."""
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    slug = slug.replace(".html", "").split("?")[0]
    return slug.replace("-", " ").strip()


def classify_track(url: str, tracks: dict) -> str:
    for name, pattern in tracks.items():
        if pattern in url:
            return name
    return "other"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float((a / np.linalg.norm(a)) @ (b / np.linalg.norm(b)))


def cosine_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query / np.linalg.norm(query)
    m = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return m @ q


def sample_holdout_urls(
    train_df: pd.DataFrame,
    tracks: dict,
    per_track: dict[str, int],
    min_paraphrases: int,
    seed: int,
) -> list[str]:
    """Stratified sample of held-out URLs, biased toward multi-paraphrase URLs."""
    rng = random.Random(seed)
    train_df = train_df.copy()
    train_df["track"] = train_df["URL"].apply(lambda u: classify_track(u, tracks))

    selected = []
    for track, n in per_track.items():
        sub = train_df[train_df["track"] == track]
        url_counts = sub["URL"].value_counts()
        multi = url_counts[url_counts >= min_paraphrases].index.tolist()
        single = url_counts[url_counts < min_paraphrases].index.tolist()
        rng.shuffle(multi)
        rng.shuffle(single)
        picks = (multi + single)[:n]
        selected.extend(picks)
    return selected


@dataclass
class HoldoutCluster:
    """A held-out URL plus all the real human posts written for it."""
    url: str
    audience_track: str
    title: str
    real_posts: list[dict] = field(default_factory=list)  # list of {"postId", "text_clean", "embedding": np.ndarray}


def build_clusters(holdout_urls: list[str], train_df: pd.DataFrame, meta: dict,
                   train_vecs: np.ndarray, tracks: dict) -> list[HoldoutCluster]:
    """Assemble HoldoutCluster objects with real posts + their embeddings."""
    train_text_by_pid = dict(zip(train_df.postId, train_df.postOriginal))
    rows_by_pid = {r["postId"]: i for i, r in enumerate(meta["train"])}

    clusters = []
    for url in holdout_urls:
        rows = train_df[train_df["URL"] == url]
        real_posts = []
        for r in rows.itertuples():
            embed_idx = rows_by_pid[int(r.postId)]
            real_posts.append({
                "postId": int(r.postId),
                "text_raw": train_text_by_pid[int(r.postId)],
                "text_clean": clean_example(train_text_by_pid[int(r.postId)]),
                "embedding": train_vecs[embed_idx],
            })
        clusters.append(HoldoutCluster(
            url=url,
            audience_track=classify_track(url, tracks),
            title=url_to_title(url),
            real_posts=real_posts,
        ))
    return clusters


def embed_via_voyage(client: voyageai.Client, texts: list[str], input_type: str, model: str) -> np.ndarray:
    BATCH = 128
    out = []
    for i in range(0, len(texts), BATCH):
        res = client.embed(texts=texts[i:i + BATCH], model=model, input_type=input_type)
        out.extend(res.embeddings)
    return np.array(out, dtype=np.float32)


def compute_metrics_for_gen_post(gen_vec: np.ndarray, cluster: HoldoutCluster) -> dict:
    """gen_vs_best_real and gen_vs_centroid for one generated post against its URL's cluster."""
    real_vecs = np.array([rp["embedding"] for rp in cluster.real_posts])
    sims = cosine_matrix(gen_vec, real_vecs)
    centroid = real_vecs.mean(axis=0)
    return {
        "gen_vs_best_real": float(sims.max()),
        "gen_vs_best_real_postId": int(cluster.real_posts[int(sims.argmax())]["postId"]),
        "gen_vs_centroid": cosine(gen_vec, centroid),
        "real_pool_size": len(cluster.real_posts),
    }


def compute_inter_human_ceiling(cluster: HoldoutCluster) -> Optional[float]:
    """Average pairwise cosine between real posts in the cluster. None if only 1 real."""
    n = len(cluster.real_posts)
    if n < 2:
        return None
    vecs = np.array([rp["embedding"] for rp in cluster.real_posts])
    norms = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sim_matrix = norms @ norms.T
    iu = np.triu_indices(n, k=1)
    return float(sim_matrix[iu].mean())


@dataclass
class GenAttempt:
    cluster_url: str
    variant_id: int
    flavor: Optional[str]
    post: str
    voice_match: int
    on_topic: int
    hook_strength: int
    compliance: str
    length_chars: int
    retries: int
    flagged: bool
    embedding: Optional[np.ndarray] = None


def run_holdout_eval(
    holdout_urls: list[str],
    n_variants: int,
    flavor_names: Optional[list[str]],
    output_prefix: Path,
    gen_model_override: Optional[str] = None,
    judge_model_override: Optional[str] = None,
):
    load_dotenv(ENV_PATH)
    config = load_config()
    gen_model = gen_model_override or config["models"]["generation"]
    judge_model = judge_model_override or config["models"]["judge"]
    embed_model = config["models"]["embedding"]
    length_hint = config["generation"]["length_hint"]
    max_tokens = config["generation"]["max_tokens"]
    temperature = config["generation"]["temperature"]
    retrieval_k = config["generation"]["retrieval_k"]
    judge_cfg = config["judge"]
    calibration_examples = judge_cfg["calibration_examples"]
    length_min = judge_cfg["length_min"]
    length_max = judge_cfg["length_max"]
    max_retries = judge_cfg["max_retries"]
    tracks = config["audience_tracks"]
    flavors_pool = resolve_flavor_pool(flavor_names, config)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(length_hint=length_hint)

    # Load train + embeddings
    train_df = pd.read_excel(TRAIN_XLSX, sheet_name="nearShoreTrain")
    retriever = Retriever()
    train_vecs = retriever.train_vecs
    meta = retriever.meta

    holdout_url_set = set(holdout_urls)
    clusters = build_clusters(holdout_urls, train_df, meta, train_vecs, tracks)

    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    # Embed slug-titles as queries (for retrieval against the remaining train pool)
    titles = [c.title for c in clusters]
    print(f"Embedding {len(titles)} held-out slug-titles as Voyage queries...")
    title_query_vecs = embed_via_voyage(voyage_client, titles, "query", embed_model)

    # GENERATE
    print(f"\n=== Generation pass ===")
    gen_attempts: list[GenAttempt] = []
    for cluster, query_vec in zip(clusters, title_query_vecs):
        print(f"\n  [{cluster.audience_track}] {cluster.title}  (real pool: {len(cluster.real_posts)})")
        examples = retriever.retrieve_by_vector(
            query_vec=query_vec,
            query_track=cluster.audience_track,
            k=retrieval_k,
            same_track_only=True,
            dedupe_by_url=True,
            exclude_urls=holdout_url_set,
        )
        article = {
            "url": cluster.url,
            "title": cluster.title,
            "audience_track": cluster.audience_track,
            "body": "",  # holdout eval: no body, model relies on examples + title
        }
        flavors_for_variants = select_flavors(n_variants, flavors_pool)
        for variant_idx, flavor in enumerate(flavors_for_variants, start=1):
            qg = generate_with_quality_gate(
                client=anthropic_client,
                gen_model=gen_model,
                judge_model=judge_model,
                system_prompt=system_prompt,
                article=article,
                examples=examples,
                calibration_examples=calibration_examples,
                max_article_chars=None,
                flavor=flavor,
                include_disclosure=False,
                disclosure_url=None,
                temperature=temperature,
                max_tokens=max_tokens,
                length_min=length_min,
                length_max=length_max,
                max_retries=max_retries,
            )
            attempt = GenAttempt(
                cluster_url=cluster.url,
                variant_id=variant_idx,
                flavor=flavor["name"] if flavor else None,
                post=qg.post,
                voice_match=qg.judgment.voice_match,
                on_topic=qg.judgment.on_topic,
                hook_strength=qg.judgment.hook_strength,
                compliance=qg.judgment.compliance,
                length_chars=qg.length.char_count,
                retries=qg.retries_used,
                flagged=qg.flagged,
            )
            gen_attempts.append(attempt)
            print(f"    v{variant_idx} [{attempt.flavor or '-':<14}] "
                  f"chars={attempt.length_chars} compl={attempt.compliance} "
                  f"voice/topic/hook={attempt.voice_match}/{attempt.on_topic}/{attempt.hook_strength}")

    # Embed generated posts (documents — same role as train posts in our embedding space)
    print(f"\nEmbedding {len(gen_attempts)} generated posts...")
    gen_vecs = embed_via_voyage(voyage_client, [a.post for a in gen_attempts], "document", embed_model)
    for attempt, vec in zip(gen_attempts, gen_vecs):
        attempt.embedding = vec

    # Judge real posts (for side-by-side comparison)
    print(f"\n=== Judging real held-out posts (for comparison baseline) ===")
    real_judgments: dict[int, dict] = {}  # postId -> {"voice_match", "on_topic", "hook_strength", "compliance", "notes"}
    for cluster in clusters:
        # Build a synthetic article dict for the judge call (judge needs article context)
        article_for_judge = {
            "url": cluster.url,
            "title": cluster.title,
            "audience_track": cluster.audience_track,
            "body": "",
        }
        # Retrieve voice examples (excluding the held-out URLs, so judge sees same context as gen)
        # Reuse the same query that gen used
        idx = clusters.index(cluster)
        examples = retriever.retrieve_by_vector(
            query_vec=title_query_vecs[idx],
            query_track=cluster.audience_track,
            k=retrieval_k,
            same_track_only=True,
            dedupe_by_url=True,
            exclude_urls=holdout_url_set,
        )
        for rp in cluster.real_posts:
            j = judge_post(
                client=anthropic_client,
                model=judge_model,
                article=article_for_judge,
                voice_examples=examples,
                calibration_examples=calibration_examples,
                post=rp["text_clean"],
            )
            real_judgments[rp["postId"]] = {
                "voice_match": j.voice_match,
                "on_topic": j.on_topic,
                "hook_strength": j.hook_strength,
                "compliance": j.compliance,
                "compliance_notes": j.compliance_notes,
                "notes": j.overall_notes,
            }
            print(f"  real postId={rp['postId']} ({cluster.audience_track}): "
                  f"voice/topic/hook={j.voice_match}/{j.on_topic}/{j.hook_strength} compl={j.compliance}")

    # COMPUTE METRICS
    print(f"\n=== Computing metrics ===")
    rows = []
    cluster_by_url = {c.url: c for c in clusters}
    for attempt in gen_attempts:
        cluster = cluster_by_url[attempt.cluster_url]
        m = compute_metrics_for_gen_post(attempt.embedding, cluster)
        rows.append({
            "url": attempt.cluster_url,
            "title": cluster.title,
            "audience_track": cluster.audience_track,
            "variant_id": attempt.variant_id,
            "flavor": attempt.flavor or "",
            "generated_post": attempt.post,
            "gen_vs_best_real": m["gen_vs_best_real"],
            "gen_vs_centroid": m["gen_vs_centroid"],
            "matched_real_postId": m["gen_vs_best_real_postId"],
            "real_pool_size": m["real_pool_size"],
            "voice_match": attempt.voice_match,
            "on_topic": attempt.on_topic,
            "hook_strength": attempt.hook_strength,
            "compliance": attempt.compliance,
            "length_chars": attempt.length_chars,
            "judge_retries": attempt.retries,
            "judge_flagged": attempt.flagged,
        })

    # Inter-human ceiling per cluster
    ceilings = {c.url: compute_inter_human_ceiling(c) for c in clusters}
    multi_paraphrase_ceiling = np.mean([v for v in ceilings.values() if v is not None]) if any(v is not None for v in ceilings.values()) else None

    # Aggregate real-vs-gen judge comparison
    real_judges_by_url: dict[str, list[dict]] = defaultdict(list)
    for cluster in clusters:
        for rp in cluster.real_posts:
            real_judges_by_url[cluster.url].append(real_judgments[rp["postId"]])

    def avg(values: list) -> Optional[float]:
        return float(np.mean(values)) if values else None

    df = pd.DataFrame(rows)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_name(output_prefix.name + ".csv")
    md_path = output_prefix.with_name(output_prefix.name + ".md")
    df.to_csv(csv_path, index=False)

    # Build markdown summary
    gen_voice = avg(df["voice_match"].tolist())
    gen_topic = avg(df["on_topic"].tolist())
    gen_hook = avg(df["hook_strength"].tolist())
    real_voice = avg([r["voice_match"] for rs in real_judges_by_url.values() for r in rs])
    real_topic = avg([r["on_topic"] for rs in real_judges_by_url.values() for r in rs])
    real_hook = avg([r["hook_strength"] for rs in real_judges_by_url.values() for r in rs])
    real_compl_fail = sum(1 for rs in real_judges_by_url.values() for r in rs if r["compliance"] == "fail")
    gen_compl_fail = sum(1 for row in rows if row["compliance"] == "fail")
    n_real = sum(len(rs) for rs in real_judges_by_url.values())

    md = [f"# Held-out evaluation — {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}", ""]
    md.append(f"- Held-out URLs: {len(clusters)}")
    md.append(f"- Per-URL variants generated: {n_variants}")
    md.append(f"- Total generated posts: {len(rows)}")
    md.append(f"- Real human posts judged (held-out): {n_real}")
    md.append(f"- Gen model: `{gen_model}`  /  Judge model: `{judge_model}`  /  Embed model: `{embed_model}`")
    md.append("")
    md.append("## Embedding-based similarity")
    md.append(f"- **gen_vs_best_real** (avg across all gen posts): **{df['gen_vs_best_real'].mean():.3f}**")
    md.append(f"- **gen_vs_centroid** (avg): {df['gen_vs_centroid'].mean():.3f}")
    if multi_paraphrase_ceiling is not None:
        md.append(f"- **inter-human ceiling** (avg pairwise cosine within multi-paraphrase clusters): **{multi_paraphrase_ceiling:.3f}**")
        gap = multi_paraphrase_ceiling - df['gen_vs_best_real'].mean()
        md.append(f"- gap (ceiling - gen_vs_best_real): {gap:+.3f}")
    md.append("")
    md.append("## Judge scores: generated vs real human posts")
    md.append("| Metric        | Generated (avg) | Real held-out (avg) | Gap   |")
    md.append("|---------------|-----------------|---------------------|-------|")
    md.append(f"| voice_match   | {gen_voice:.2f}            | {real_voice:.2f}                | {gen_voice - real_voice:+.2f} |")
    md.append(f"| on_topic      | {gen_topic:.2f}            | {real_topic:.2f}                | {gen_topic - real_topic:+.2f} |")
    md.append(f"| hook_strength | {gen_hook:.2f}            | {real_hook:.2f}                | {gen_hook - real_hook:+.2f} |")
    md.append(f"| compliance fails | {gen_compl_fail}/{len(rows)} | {real_compl_fail}/{n_real} | — |")
    md.append("")
    md.append("## Per-cluster results")
    md.append("| URL slug | track | real pool | gen_vs_best (avg) | inter-human ceiling |")
    md.append("|----------|-------|-----------|-------------------|---------------------|")
    for c in clusters:
        urlslug = c.url.rsplit("/", 1)[-1].replace(".html", "")[:60]
        gen_for_url = df[df["url"] == c.url]
        avg_best = gen_for_url["gen_vs_best_real"].mean()
        ceiling = ceilings.get(c.url)
        ceiling_str = f"{ceiling:.3f}" if ceiling is not None else "n/a (single real)"
        md.append(f"| `{urlslug}` | {c.audience_track} | {len(c.real_posts)} | {avg_best:.3f} | {ceiling_str} |")
    md.append("")

    md_path.write_text("\n".join(md))
    print(f"\n=== Eval complete ===")
    print(f"  csv: {csv_path}")
    print(f"  md:  {md_path}")
    return rows
