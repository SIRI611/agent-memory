"""Rank the target among stored memories using cached query-time Jev scores.

Two pools:
- inmind: target plus the 238 background user turns (the InMind setting).
- multi_fact: that pool plus the other 121 unique target facts as competing user facts.
Ties are broken by memory hash so ranks are deterministic.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter

from jev_persist.corpus import ROOT, load_config
from jev_persist.query_relevance import load_cache, pair_key, pools

OUT = ROOT / "results" / "query_relevance" / "summary.json"
KS = (1, 3, 5, 10)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    tokens = [tokenize(d) for d in docs]
    avg = sum(len(t) for t in tokens) / len(tokens)
    df = Counter(term for t in tokens for term in set(t))
    n = len(docs)
    scores = []
    for t in tokens:
        tf = Counter(t)
        score = 0.0
        for term in set(tokenize(query)):
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(t) / avg))
        scores.append(score)
    return scores


def rank_of(target_hash: str, scored: list[tuple[float, str]]) -> int:
    ordered = sorted(scored, key=lambda row: (-row[0], row[1]))
    return [h for _, h in ordered].index(target_hash) + 1


def main() -> None:
    config = load_config()
    targets, memories = pools(config)
    cache = load_cache()
    background = [m for m in memories if m["kind"] == "background"]
    other_targets = [m for m in memories if m["kind"] == "target"]
    result = {}
    per_task = []
    for pool_name in ("inmind", "multi_fact"):
        ranks, bm25_ranks, above_half, n_selected, target_p = [], [], [], [], []
        missing = 0
        for task in targets:
            query = task["query"]
            pool = list(background)
            if pool_name == "multi_fact":
                pool += [m for m in other_targets if m["text_hash"] != task["text_hash"]]
            pool.append({"text_hash": task["text_hash"], "text": task["user_message"]})
            keys = [pair_key(query, m["text_hash"]) for m in pool]
            if any(key not in cache for key in keys):
                missing += 1
                continue
            scored = [(cache[key], m["text_hash"]) for key, m in zip(keys, pool)]
            rank = rank_of(task["text_hash"], scored)
            lexical = bm25_scores(query, [m["text"] for m in pool])
            bm25_ranks.append(rank_of(task["text_hash"], list(zip(lexical, [m["text_hash"] for m in pool]))))
            p = cache[pair_key(query, task["text_hash"])]
            ranks.append(rank)
            target_p.append(p)
            above_half.append(p > 0.5)
            n_selected.append(sum(score > 0.5 for score, _ in scored))
            if pool_name == "multi_fact":
                per_task.append({"task_id": task["task_id"], "rank_multi": rank, "p": p})
        n = len(ranks)
        if not n:
            result[pool_name] = {"n": 0, "missing": missing}
            continue
        result[pool_name] = {
            "n": n,
            "missing": missing,
            "pool_size": len(background) + (len(other_targets) - 1 if pool_name == "multi_fact" else 0) + 1,
            "median_rank": statistics.median(ranks),
            "mean_rank": round(statistics.mean(ranks), 2),
            **{f"recall@{k}": round(sum(r <= k for r in ranks) / n, 3) for k in KS},
            "target_p_over_0.5": round(sum(above_half) / n, 3),
            "median_items_over_0.5": statistics.median(n_selected),
            "mean_target_p": round(statistics.mean(target_p), 3),
            "bm25_median_rank": statistics.median(bm25_ranks),
            **{f"bm25_recall@{k}": round(sum(r <= k for r in bm25_ranks) / n, 3) for k in KS},
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": result, "per_task": per_task}, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
