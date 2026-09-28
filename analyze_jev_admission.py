"""Rank InMind targets against background turns using cached Jev scores.

Weights are the pre-specified v0.1 formula. They are not fit on this set.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "results" / "jev_admission_cache.jsonl"
TASKS = ROOT / "data" / "inmind.jsonl"
SUMMARY = ROOT / "results" / "jev_admission_summary.json"

WEIGHTS = {
    "constraint": 0.35,
    "impact": 0.30,
    "stable": 0.20,
    "user_specific": 0.15,
}
BUDGETS = (128, 256, 512, 1024, 2048)
KEYWORDS = (
    "allerg",
    "asthma",
    "diagnos",
    "pregnan",
    "medication",
    "medicine",
    "halal",
    "kosher",
    "vegan",
    "vegetarian",
    "diabet",
    "wheelchair",
    "hearing",
    "gluten",
    "lactose",
    "seizure",
    "disability",
    "injur",
    "surgery",
    "budget",
    "debt",
    "visa",
    "custody",
)


def auroc(labels: list[int], scores: list[float]) -> float | None:
    pairs = [(score, label) for score, label in zip(scores, labels) if label in (0, 1)]
    pos = [score for score, label in pairs if label == 1]
    neg = [score for score, label in pairs if label == 0]
    if not pos or not neg:
        return None
    wins = ties = 0.0
    for left in pos:
        for right in neg:
            if left > right:
                wins += 1
            elif left == right:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def average_precision(labels: list[int], scores: list[float]) -> float | None:
    order = sorted(range(len(labels)), key=lambda i: scores[i], reverse=True)
    hit = 0
    total = 0.0
    positives = sum(labels)
    if positives == 0:
        return None
    for rank, index in enumerate(order, start=1):
        if labels[index] == 1:
            hit += 1
            total += hit / rank
    return total / positives


def features(row: dict) -> dict[str, float]:
    answers = row["answers"]
    type_probs = answers["memory_type"].get("probabilities") or {}
    impact = answers["impact"]
    score = float(impact["score"])
    levels = max(len(impact.get("probabilities") or {}), 1)
    return {
        "constraint": float(type_probs.get("stable_constraint", 0.0)),
        "stable": float(answers["stable"]["probability"]),
        "decision_relevant": float(answers["decision_relevant"]["probability"]),
        "impact": score / (levels - 1 if levels > 1 else 1),
        "user_specific": float(answers["user_specific"]["probability"]),
        "choice_confidence": float(answers["memory_type"].get("confidence") or 0.0),
    }


def priority(parts: dict[str, float]) -> float:
    return sum(WEIGHTS[name] * parts[name] for name in WEIGHTS)


def keyword_score(text: str) -> float:
    lowered = text.lower()
    return float(sum(token in lowered for token in KEYWORDS))


def token_estimate(char_len: int) -> int:
    return max(1, math.ceil(char_len / 4))


def pack_hit(ranked: list[dict], budget: int) -> bool:
    used = 0
    for item in ranked:
        cost = token_estimate(item["char_len"])
        if used + cost > budget:
            continue
        if item["label"] == "target":
            return True
        used += cost
    return False


def main() -> None:
    rows = [json.loads(line) for line in CACHE.read_text().splitlines() if line.strip()]
    ok = [row for row in rows if row.get("ok")]
    by_hash = {row["text_hash"]: row for row in ok}
    tasks = [
        json.loads(line)
        for line in TASKS.read_text().splitlines()
        if line.strip()
    ]
    import hashlib
    import re

    def key_for(text: str) -> str:
        payload = f"v0.1-admission\ntypesafe-ai/jev\n{text}"
        return hashlib.sha256(payload.encode()).hexdigest()

    targets = []
    for task in tasks:
        text = re.split(r"\n---TIMESTAMP:", task["user_message"], maxsplit=1)[0].strip()
        row = by_hash[key_for(text)]
        parts = features(row)
        targets.append(
            {
                "task_id": task["task_id"],
                "domain": task["domain"],
                "text": text,
                "label": "target",
                "char_len": len(text),
                "priority": priority(parts),
                "keyword": keyword_score(text),
                "shortness": -len(text),
                **parts,
            }
        )
    background = []
    for row in ok:
        if row["label"] != "background":
            continue
        parts = features(row)
        background.append(
            {
                "label": "background",
                "char_len": row["char_len"],
                "priority": priority(parts),
                "keyword": 0.0,
                "shortness": -row["char_len"],
                **parts,
            }
        )
    # Keyword baseline needs the raw text. Recover it from the cache only if stored.
    # The cache does not store raw text, so keyword scores for background are computed
    # by re-reading the background file below.
    background_rows = {
        row["text_hash"]: row for row in ok if row["label"] == "background"
    }
    raw_bg = [
        json.loads(line)
        for line in (ROOT / "data" / "lme_s_background.jsonl").read_text().splitlines()
        if line.strip()
    ]
    background = []
    seen = set()
    for turn in raw_bg:
        if turn["role"] != "user":
            continue
        text = re.split(r"\n---TIMESTAMP:", turn["content"], maxsplit=1)[0].strip()
        text_hash = key_for(text)
        if text_hash in seen or text_hash not in background_rows:
            continue
        seen.add(text_hash)
        row = background_rows[text_hash]
        parts = features(row)
        background.append(
            {
                "label": "background",
                "text": text,
                "char_len": len(text),
                "priority": priority(parts),
                "keyword": keyword_score(text),
                "shortness": -len(text),
                **parts,
            }
        )

    pool_labels = [1] * len(targets) + [0] * len(background)
    names = [
        "priority",
        "constraint",
        "decision_relevant",
        "stable",
        "impact",
        "user_specific",
        "keyword",
        "shortness",
    ]
    metrics = {}
    for name in names:
        scores = [item[name] for item in targets] + [item[name] for item in background]
        metrics[name] = {
            "auroc": auroc(pool_labels, scores),
            "average_precision": average_precision(pool_labels, scores),
        }

    rng = random.Random(0)
    rank_stats = {name: [] for name in ("priority", "decision_relevant", "keyword", "shortness")}
    budget_hits = {name: {budget: 0 for budget in BUDGETS} for name in rank_stats}
    for target in targets:
        for name in rank_stats:
            ranked = sorted(
                [target, *background],
                key=lambda item: (item[name], rng.random()),
                reverse=True,
            )
            rank = next(i for i, item in enumerate(ranked, start=1) if item is target)
            rank_stats[name].append(rank)
            for budget in BUDGETS:
                if pack_hit(ranked, budget):
                    budget_hits[name][budget] += 1

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    n = len(targets)
    summary = {
        "n_targets": n,
        "n_background": len(background),
        "n_scored": len(ok),
        "n_failed": sum(1 for row in rows if not row.get("ok")),
        "cost_sum": sum(float(row.get("cost") or 0) for row in ok),
        "weights": WEIGHTS,
        "note": "Query and bridge text were not shown to Jev. Token budgets use ceil(chars/4), not the answer-model tokenizer.",
        "separation": metrics,
        "median_rank": {name: sorted(values)[len(values) // 2] for name, values in rank_stats.items()},
        "mean_rank": {name: mean(values) for name, values in rank_stats.items()},
        "budget_recall": {
            name: {str(budget): hits / n for budget, hits in buckets.items()}
            for name, buckets in budget_hits.items()
        },
        "target_priority_mean": mean([item["priority"] for item in targets]),
        "background_priority_mean": mean([item["priority"] for item in background]),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
