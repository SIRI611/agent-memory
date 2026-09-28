"""Summarize end-to-end answers and judgements. No model calls."""

from __future__ import annotations

import json
import random

from jev_persist.corpus import ROOT
from jev_persist.e2e import ANSWERS, CONDITIONS, JUDGES, NEGATIVE_CONDITIONS, load_rows

OUT = ROOT / "results" / "e2e" / "report.json"
PAIRED = [
    ("jev_top5", "bm25_top5"),
    ("jev_top5", "embed_top5"),
    ("jev_top5", "jevmem_style_top5"),
    ("jev_top5", "jev_priority_512"),
    ("jev_top5", "always_in_state"),
    ("two_stage_top5", "jev_top5"),
    ("oracle", "jev_top5"),
]


def ci(values: list[int], rounds: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(rounds))
    return round(means[int(0.025 * rounds)], 3), round(means[int(0.975 * rounds)], 3)


def main() -> None:
    answers = load_rows(ANSWERS)
    judges = load_rows(JUDGES)
    scores: dict[tuple[str, str], dict[int, int]] = {}
    for row in judges.values():
        if "query_task" in row or row.get("score") is None:
            continue
        scores.setdefault((row["condition"], row["name"]), {})[row["task_id"]] = row["score"]
    visible = {}
    for row in answers.values():
        if "query_task" in row or row["name"] != "query":
            continue
        visible.setdefault(row["condition"], {})[row["task_id"]] = int(bool(row["target_visible"]))
    table = {}
    for condition in CONDITIONS:
        entry = {}
        for name in ("naive", "application", "answer_only"):
            values = list(scores.get((condition, name), {}).values())
            if values:
                entry[name] = {"n": len(values), "acc": round(sum(values) / len(values), 3), "ci95": ci(values)}
        vis = list(visible.get(condition, {}).values())
        if vis:
            entry["target_in_context"] = round(sum(vis) / len(vis), 3)
        table[condition] = entry
    paired = {}
    for a, b in PAIRED:
        for name in ("application", "answer_only"):
            sa, sb = scores.get((a, name), {}), scores.get((b, name), {})
            common = sorted(set(sa) & set(sb))
            if not common:
                continue
            diffs = [sa[t] - sb[t] for t in common]
            paired[f"{a} - {b} [{name}]"] = {
                "n": len(common),
                "diff": round(sum(diffs) / len(diffs), 3),
                "ci95": ci(diffs),
                "a_only": sum(d == 1 for d in diffs),
                "b_only": sum(d == -1 for d in diffs),
            }
    negatives = {}
    for condition in NEGATIVE_CONDITIONS:
        rows = [r for r in judges.values() if r.get("query_task") is not None and r["condition"] == condition
                and "mentioned" in r and "relevant" in r]
        if not rows:
            continue
        irrelevant = [r for r in rows if r["relevant"] == 0]
        negatives[condition] = {
            "n": len(rows),
            "judged_irrelevant": len(irrelevant),
            "mentioned_when_irrelevant": round(sum(r["mentioned"] for r in irrelevant) / max(1, len(irrelevant)), 3),
            "mentioned_overall": round(sum(r["mentioned"] for r in rows) / len(rows), 3),
        }
    report = {"conditions": table, "paired": paired, "negative_controls": negatives}
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
