"""Per-query memory selection for every compared method.

All methods see the same pool of stored user turns. None of them sees the
bridge explanation. Frozen choices (set before any answer was scored):
- K = 5 for every top-k method (InMind Naive RAG uses top-5).
- Jev-Mem-style anchors: reciprocal-rank fusion of BM25 and embedding, kappa 60,
  top 20 anchors, then Jev reranks the anchors. No graph expansion is simulated.
- Two-stage gate: keep a memory at write time when P(user_specific) >= 0.5.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache

from jev_persist.analyze_query_relevance import bm25_scores
from jev_persist.corpus import ROOT, load_config, scored_items
from jev_persist.pack import pack
from jev_persist.query_relevance import load_cache, pair_key, sha

K = 5
ANCHORS = 20
RRF_KAPPA = 60
GATE = 0.5
EMBED_PATH = ROOT / "results" / "query_relevance" / "embed_minilm.json"
MULTI_FACT_N = 15
MULTI_FACT_PER_DOMAIN = 2


@lru_cache(maxsize=1)
def corpus() -> tuple[list[dict], list[dict], dict]:
    config = load_config()
    targets, background = scored_items(config)
    return targets, background, config


@lru_cache(maxsize=1)
def jev_cache() -> dict:
    return load_cache()


@lru_cache(maxsize=1)
def embed_sims() -> dict:
    if not EMBED_PATH.exists():
        return {}
    return json.loads(EMBED_PATH.read_text())["sims"]


def memory_text(item: dict) -> str:
    return item.get("text") or item["user_message"]


def as_memory(item: dict) -> dict:
    return {**item, "text": memory_text(item)}


def unique_targets() -> list[dict]:
    targets, _, _ = corpus()
    seen = {}
    for item in targets:
        seen.setdefault(item["text_hash"], item)
    return list(seen.values())


def multi_fact_distractors(task: dict) -> list[dict]:
    """Other InMind facts from other domains, at most two per domain, seeded by task id."""
    rng = random.Random(f"multi-fact-{task['task_id']}")
    others = [
        t for t in unique_targets()
        if t["text_hash"] != task["text_hash"] and t["domain"] != task["domain"]
        and (t["entity_1"] or "").lower() != (task["entity_1"] or "-").lower()
    ]
    rng.shuffle(others)
    picked, per_domain = [], {}
    for item in others:
        if per_domain.get(item["domain"], 0) >= MULTI_FACT_PER_DOMAIN:
            continue
        picked.append(item)
        per_domain[item["domain"]] = per_domain.get(item["domain"], 0) + 1
        if len(picked) == MULTI_FACT_N:
            break
    return picked


def pool_for(task: dict, pool: str = "inmind", stored_target: dict | None = None) -> list[dict]:
    """Stored memories. stored_target overrides which fact is injected (negative controls)."""
    _, background, _ = corpus()
    fact = stored_target or task
    items = [as_memory(b) for b in background] + [as_memory(fact)]
    if pool == "multi_fact":
        items += [as_memory(t) for t in multi_fact_distractors(fact)]
    return items


def ordered(scores: list[float], items: list[dict]) -> list[dict]:
    return [item for _, item in sorted(zip(scores, items), key=lambda row: (-row[0], row[1]["text_hash"]))]


def rank_bm25(query: str, items: list[dict]) -> list[dict]:
    return ordered(bm25_scores(query, [i["text"] for i in items]), items)


def rank_embed(query: str, items: list[dict]) -> list[dict]:
    sims = embed_sims().get(sha(query))
    if sims is None:
        raise KeyError("embedding similarities missing; run jev_persist.embed_scores")
    return ordered([sims[i["text_hash"]] for i in items], items)


def jev_score(query: str, item: dict) -> float:
    return jev_cache()[pair_key(query, item["text_hash"])]


def rank_jev(query: str, items: list[dict]) -> list[dict]:
    return ordered([jev_score(query, i) for i in items], items)


def rank_jevmem_style(query: str, items: list[dict], anchors: int = ANCHORS) -> list[dict]:
    fused = {}
    for ranking in (rank_bm25(query, items), rank_embed(query, items)):
        for position, item in enumerate(ranking, start=1):
            fused[item["text_hash"]] = fused.get(item["text_hash"], 0.0) + 1.0 / (RRF_KAPPA + position)
    anchor_items = ordered([fused[i["text_hash"]] for i in items], items)[:anchors]
    return rank_jev(query, anchor_items)


def rank_two_stage(query: str, items: list[dict], gate: float = GATE) -> list[dict]:
    kept = [i for i in items if float(i["user_specific"]) >= gate]
    return rank_jev(query, kept)


def write_time_core(items: list[dict], budget: int = 512) -> list[dict]:
    return pack(items, "jev_priority", budget)["chosen"]


RANKERS = {
    "bm25": rank_bm25,
    "embed": rank_embed,
    "jev": rank_jev,
    "jevmem_style": rank_jevmem_style,
    "two_stage": rank_two_stage,
}


def select(method: str, query: str, items: list[dict], k: int = K) -> list[dict]:
    if method == "write_v01_512":
        return write_time_core(items)
    return RANKERS[method](query, items)[:k]
