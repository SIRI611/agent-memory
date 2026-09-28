"""Offline retrieval metrics for write-time units. No model calls.

Recall@k: a task counts when any unit from its target turn is in the top k.
Negative inclusion: for each cross-domain pair (stored fact, unrelated query),
whether a unit from the stored target turn enters the top 5.
Threshold sweep (P3): keep at most 5 units with score >= tau. InMind is the
development set, so a tau chosen here must be confirmed on held-out data.
"""

from __future__ import annotations

import json
import statistics

from jev_persist import units_routing as ur
from jev_persist.corpus import ROOT
from jev_persist.offline_report import negative_partners
from jev_persist.routing import corpus

OUT = ROOT / "results" / "v2" / "offline.json"
METHODS = ["facts", "rules", "rules_max", "app_only"]
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def planned(query: str) -> str:
    from jev_persist.query_plan import planned_state

    return planned_state(query)


def target_rank(method: str, state: str, task: dict, units: list[dict]) -> tuple[int | None, float]:
    rows = ur.ranked(method, state, units)
    for position, (score, unit) in enumerate(rows, start=1):
        if unit["turn_hash"] == task["text_hash"]:
            return position, score
    return None, 0.0


def evaluate(method: str, use_plan: bool = False) -> dict | None:
    targets, _, _ = corpus()
    by_id = {t["task_id"]: t for t in targets}
    try:
        pos, neg = [], []
        for task in targets:
            state = planned(task["query"]) if use_plan else task["query"]
            rows = ur.ranked(method, state, ur.unit_pool(task))
            pos.append((rows, task))
        for stored_id, query_id in negative_partners().items():
            stored, asked = by_id[stored_id], by_id[query_id]
            state = planned(asked["query"]) if use_plan else asked["query"]
            neg.append((ur.ranked(method, state, ur.unit_pool(stored, stored_target=stored)), stored))
    except KeyError:
        return None

    def rank_of(rows, task):
        for position, (score, unit) in enumerate(rows, start=1):
            if unit["turn_hash"] == task["text_hash"]:
                return position, score
        return None, 0.0

    ranks = [rank_of(rows, task) for rows, task in pos]
    neg_ranks = [rank_of(rows, task) for rows, task in neg]
    n = len(ranks)
    out = {
        "recall@1": round(sum(1 for r, _ in ranks if r and r <= 1) / n, 3),
        "recall@5": round(sum(1 for r, _ in ranks if r and r <= 5) / n, 3),
        "recall@10": round(sum(1 for r, _ in ranks if r and r <= 10) / n, 3),
        "median_target_score": round(statistics.median(s for _, s in ranks), 3),
        "neg_in_top5": round(sum(1 for r, _ in neg_ranks if r and r <= 5) / len(neg_ranks), 3),
        "median_neg_score": round(statistics.median(s for _, s in neg_ranks), 3),
        "misses@5": sorted(task["task_id"] for (r, _), (_, task) in zip(ranks, pos) if not (r and r <= 5)),
    }
    sweep = {}
    for tau in TAUS:
        kept = [sum(1 for s, _ in rows[:5] if s >= tau) for rows, _ in pos]
        neg_kept = [sum(1 for s, _ in rows[:5] if s >= tau) for rows, _ in neg]
        sweep[str(tau)] = {
            "recall": round(sum(1 for (r, s) in ranks if r and r <= 5 and s >= tau) / n, 3),
            "neg_inclusion": round(sum(1 for (r, s) in neg_ranks if r and r <= 5 and s >= tau) / len(neg_ranks), 3),
            "mean_units": round(sum(kept) / n, 2),
            "neg_mean_units": round(sum(neg_kept) / len(neg_kept), 2),
            "empty_context": round(sum(1 for k in kept if k == 0) / n, 3),
        }
    out["threshold_sweep"] = sweep
    return out


def main() -> None:
    report = {}
    for method in METHODS:
        report[method] = evaluate(method)
    report["plan_facts"] = evaluate("facts", use_plan=True)
    OUT.write_text(json.dumps(report, indent=2))
    for name, entry in report.items():
        if entry is None:
            print(name, "missing scores")
            continue
        brief = {k: v for k, v in entry.items() if k not in ("threshold_sweep",)}
        print(name, json.dumps(brief))


if __name__ == "__main__":
    main()
