"""Summarize round-two answers and judgements together with round one. No model calls.

Noise: oracle and two_stage_top5 were answered three times (round one plus
@r1, @r2) and the round-one answers were re-judged twice (application@j1, @j2).
Paired comparisons against those two baselines use the per-task mean over the
three answer runs.
"""

from __future__ import annotations

import json
import statistics

from jev_persist import e2e, e2e_v2
from jev_persist.corpus import ROOT
from jev_persist.e2e_report import ci

OUT = ROOT / "results" / "v2" / "report.json"
BASELINES = {"oracle": ["oracle", "oracle@r1", "oracle@r2"],
             "two_stage_top5": ["two_stage_top5", "two_stage_top5@r1", "two_stage_top5@r2"]}
PAIRED = [
    ("facts_top5", "two_stage_top5"),
    ("rules_top5", "two_stage_top5"),
    ("rules_top5", "facts_top5"),
    ("rules_top5", "rules_top5_noaction"),
    ("rules_top5_noaction", "facts_top5"),
    ("oracle_facts", "oracle"),
    ("oracle_rules", "oracle_facts"),
    ("oracle_rules", "oracle"),
    ("rules_top5", "oracle"),
    ("llm_scan_top5", "two_stage_top5"),
    ("rules_top5", "llm_scan_top5"),
    ("plan_facts_top5", "facts_top5"),
    ("jev_top5", "two_stage_top5"),
    ("two_stage_t0.1", "two_stage_top5"),
    ("two_stage_t0.2", "two_stage_top5"),
    ("fuse_top10", "two_stage_top5"),
    ("fuse_top10", "jev_top5"),
]
NEGATIVE = ["neg_oracle", "neg_jev_top5", *e2e_v2.NEGATIVE]


def load_scores() -> tuple[dict, dict, list[dict]]:
    scores: dict[tuple[str, str], dict[int, int]] = {}
    visible: dict[str, dict[int, int]] = {}
    negatives = []
    for answers_path, judges_path in ((e2e.ANSWERS, e2e.JUDGES), (e2e_v2.ANSWERS, e2e_v2.JUDGES)):
        for row in e2e.load_rows(judges_path).values():
            if row.get("query_task") is not None:
                negatives.append(row)
            elif row.get("score") is not None:
                scores.setdefault((row["condition"], row["name"]), {})[row["task_id"]] = row["score"]
        for row in e2e.load_rows(answers_path).values():
            if "query_task" not in row and row["name"] == "query":
                visible.setdefault(row["condition"], {})[row["task_id"]] = int(bool(row["target_visible"]))
    return scores, visible, negatives


def per_task(scores: dict, condition: str, name: str) -> dict[int, float]:
    runs = BASELINES.get(condition, [condition])
    found = [scores.get((run, name), {}) for run in runs]
    found = [f for f in found if f]
    if not found:
        return {}
    common = set.intersection(*(set(f) for f in found))
    return {t: sum(f[t] for f in found) / len(found) for t in common}


def main() -> None:
    scores, visible, negatives = load_scores()
    conditions = sorted({c for c, _ in scores} | set(visible))
    table = {}
    for condition in conditions:
        entry = {}
        for name in ("application", "answer_only"):
            values = list(scores.get((condition, name), {}).values())
            if values:
                entry[name] = {"n": len(values), "acc": round(sum(values) / len(values), 3), "ci95": ci(values)}
        vis = list(visible.get(condition, {}).values())
        if vis:
            entry["target_in_context"] = round(sum(vis) / len(vis), 3)
        if entry:
            table[condition] = entry

    noise = {}
    for base, runs in BASELINES.items():
        accs = [table[r]["application"]["acc"] for r in runs if "application" in table.get(r, {})]
        maps = [scores.get((r, "application"), {}) for r in runs]
        common = set.intersection(*(set(m) for m in maps if m)) if all(maps) else set()
        flips = sum(1 for t in common if len({m[t] for m in maps}) > 1)
        judge_maps = [scores.get((base, n), {}) for n in ("application", "application@j1", "application@j2")]
        jcommon = set.intersection(*(set(m) for m in judge_maps if m)) if all(judge_maps) else set()
        noise[base] = {
            "answer_run_accs": accs,
            "answer_run_sd": round(statistics.pstdev(accs), 3) if len(accs) > 1 else None,
            "tasks_flipping_across_answer_runs": f"{flips}/{len(common)}",
            "rejudge_accs": [round(sum(m.values()) / len(m), 3) for m in judge_maps if m],
            "tasks_flipping_across_judges": f"{sum(1 for t in jcommon if len({m[t] for m in judge_maps}) > 1)}/{len(jcommon)}",
            "mean_over_runs": round(statistics.mean(per_task(scores, base, 'application').values()), 3) if common else None,
        }

    paired = {}
    for a, b in PAIRED:
        for name in ("application", "answer_only"):
            sa, sb = per_task(scores, a, name), per_task(scores, b, name)
            common = sorted(set(sa) & set(sb))
            if not common:
                continue
            diffs = [sa[t] - sb[t] for t in common]
            paired[f"{a} - {b} [{name}]"] = {
                "n": len(common), "diff": round(sum(diffs) / len(diffs), 3), "ci95": ci(diffs),
                "a_better": sum(d > 0 for d in diffs), "b_better": sum(d < 0 for d in diffs),
            }

    neg_table = {}
    for condition in NEGATIVE:
        rows = [r for r in negatives if r["condition"] == condition and "mentioned" in r and "relevant" in r]
        if not rows:
            continue
        irrelevant = [r for r in rows if r["relevant"] == 0]
        neg_table[condition] = {
            "n": len(rows), "judged_irrelevant": len(irrelevant),
            "mentioned_when_irrelevant": round(sum(r["mentioned"] for r in irrelevant) / max(1, len(irrelevant)), 3),
            "mentioned_overall": round(sum(r["mentioned"] for r in rows) / len(rows), 3),
        }
    report = {"conditions": table, "noise": noise, "paired": paired, "negative_controls": neg_table}
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
