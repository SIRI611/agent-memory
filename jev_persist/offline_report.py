"""Offline routing comparison from cached scores. No model calls.

Reports, for indirect queries on all 125 tasks:
- target recall@k per method on the InMind pool and the multi-fact pool;
- Jev-Mem-style anchor sweeps and two-stage gate sweeps (descriptive only;
  the frozen settings are ANCHORS and GATE in routing.py);
- negative controls: the stored fact is paired with another task's query.
"""

from __future__ import annotations

import json
import random
import statistics

from jev_persist.corpus import ROOT
from jev_persist.routing import (
    ANCHORS,
    GATE,
    K,
    corpus,
    jev_score,
    pool_for,
    rank_jevmem_style,
    rank_two_stage,
    select,
)

OUT = ROOT / "results" / "routing"
METHODS = ("bm25", "embed", "jev", "jevmem_style", "two_stage", "write_v01_512")


def negative_partners() -> dict[int, int]:
    """Each task's stored fact is queried with a task from a different domain."""
    targets, _, _ = corpus()
    order = sorted(targets, key=lambda t: (t["domain"], t["task_id"]))
    shift = len(order) // 2
    pairs = {order[i]["task_id"]: order[(i + shift) % len(order)]["task_id"] for i in range(len(order))}
    by_id = {t["task_id"]: t for t in targets}
    if any(by_id[a]["domain"] == by_id[b]["domain"] for a, b in pairs.items()):
        raise RuntimeError("cross-domain pairing failed")
    return pairs


def visible(method: str, task: dict, pool: str, k: int = K) -> bool:
    items = pool_for(task, pool)
    chosen = select(method, task["query"], items, k)
    return any(i["text_hash"] == task["text_hash"] for i in chosen)


def recall_table(pool: str) -> dict:
    targets, _, _ = corpus()
    table = {}
    for method in METHODS:
        row = {}
        ks = (1, 5, 10) if method != "write_v01_512" else (None,)
        for k in ks:
            hits = sum(visible(method, t, pool, k or K) for t in targets)
            row["visible" if k is None else f"recall@{k}"] = round(hits / len(targets), 3)
        table[method] = row
    return table


def sweeps() -> dict:
    targets, _, _ = corpus()
    anchor = {}
    for n in (5, 10, 20, 50, 100):
        hits = 0
        for t in targets:
            ranked = rank_jevmem_style(t["query"], pool_for(t), anchors=n)[:K]
            hits += any(i["text_hash"] == t["text_hash"] for i in ranked)
        anchor[n] = round(hits / len(targets), 3)
    gate = {}
    for g in (0.3, 0.5, 0.7, 0.8):
        hits, sizes = 0, []
        for t in targets:
            items = pool_for(t)
            sizes.append(sum(float(i["user_specific"]) >= g for i in items))
            ranked = rank_two_stage(t["query"], items, gate=g)[:K]
            hits += any(i["text_hash"] == t["text_hash"] for i in ranked)
        gate[g] = {"recall@5": round(hits / len(targets), 3), "median_pool": statistics.median(sizes)}
    return {"jevmem_anchor_recall@5": anchor, "two_stage_gate": gate}


def negative_controls() -> dict:
    targets, _, _ = corpus()
    by_id = {t["task_id"]: t for t in targets}
    partners = negative_partners()
    rows = []
    for task_id, partner_id in partners.items():
        stored = by_id[task_id]
        query = by_id[partner_id]["query"]
        items = pool_for(stored, stored_target=stored)
        chosen = select("jev", query, items)
        rows.append(
            {
                "stored_task": task_id,
                "query_task": partner_id,
                "fact_in_top5": any(i["text_hash"] == stored["text_hash"] for i in chosen),
                "fact_p": jev_score(query, stored),
                "true_pair_p": jev_score(stored["query"], stored),
            }
        )
    n = len(rows)
    return {
        "n": n,
        "fact_in_jev_top5_irrelevant_query": round(sum(r["fact_in_top5"] for r in rows) / n, 3),
        "fact_p_over_0.5_irrelevant_query": round(sum(r["fact_p"] > 0.5 for r in rows) / n, 3),
        "fact_p_over_0.5_true_query": round(sum(r["true_pair_p"] > 0.5 for r in rows) / n, 3),
        "median_fact_p_irrelevant": round(statistics.median(r["fact_p"] for r in rows), 3),
        "median_fact_p_true": round(statistics.median(r["true_pair_p"] for r in rows), 3),
        "pairs": rows,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "frozen": {"k": K, "anchors": ANCHORS, "gate": GATE},
        "inmind_pool": recall_table("inmind"),
        "multi_fact_pool": recall_table("multi_fact"),
        "sweeps_descriptive": sweeps(),
    }
    negatives = negative_controls()
    report["negative_controls"] = {k: v for k, v in negatives.items() if k != "pairs"}
    (OUT / "offline_report.json").write_text(json.dumps(report, indent=2))
    (OUT / "negative_pairs.json").write_text(json.dumps(negatives["pairs"], indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
