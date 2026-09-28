"""Write the frozen pilot artifacts. This module does not call any model API."""

from __future__ import annotations

import json

from jev_persist.corpus import ROOT, load_config, scored_items
from jev_persist.render import (
    answer_calls,
    condition_items,
    dump_json,
    forbidden_in_answer_context,
    judge_calls,
)

FRAMEWORK = ROOT / "results" / "framework"


def select_pilot(targets: list[dict], config: dict) -> list[dict]:
    threshold = float(config["low_priority_threshold"])
    size = int(config["pilot_size"])
    max_low = int(config["max_low_priority_in_pilot"])
    low = sorted(
        (item for item in targets if float(item["jev_priority"]) < threshold),
        key=lambda item: (float(item["jev_priority"]), item["task_id"]),
    )
    high = sorted(
        (item for item in targets if float(item["jev_priority"]) >= threshold),
        key=lambda item: (-float(item["jev_priority"]), item["task_id"]),
    )
    chosen = list(low[:max_low])
    chosen_ids = {item["task_id"] for item in chosen}
    domains = {item["domain"] for item in chosen}
    for item in high:
        if len(chosen) >= size:
            break
        if item["domain"] in domains:
            continue
        chosen.append(item)
        chosen_ids.add(item["task_id"])
        domains.add(item["domain"])
    for item in high:
        if len(chosen) >= size:
            break
        if item["task_id"] in chosen_ids:
            continue
        chosen.append(item)
        chosen_ids.add(item["task_id"])
    if len(chosen) != size:
        raise RuntimeError(f"pilot selection produced {len(chosen)} tasks, expected {size}")
    return sorted(chosen, key=lambda item: item["task_id"])


def public_task(item: dict) -> dict:
    return {
        "task_id": item["task_id"],
        "domain": item["domain"],
        "user_message": item["user_message"],
        "naive_query": item["naive_query"],
        "query": item["query"],
        "explanation": item["explanation"],
        "jev_priority": item["jev_priority"],
        "memory_type": item["memory_type"],
        "user_specific": item["user_specific"],
        "decision_relevant": item["decision_relevant"],
        "constraint": item["constraint"],
        "impact": item["impact"],
        "stable": item["stable"],
        "tokens": item["tokens"],
        "text_hash": item["text_hash"],
    }


def build() -> dict:
    config = load_config()
    targets, background = scored_items(config)
    pilot = select_pilot(targets, config)
    threshold = float(config["low_priority_threshold"])
    failures = [
        public_task(item)
        for item in sorted(targets, key=lambda item: float(item["jev_priority"]))
        if float(item["jev_priority"]) < threshold
    ]
    packing = []
    for target in targets:
        for selector in config["offline_selectors"]:
            for budget in config["budgets"]:
                _, meta = condition_items(
                    target,
                    background,
                    {"id": "packed", "selector": selector, "budget": budget},
                )
                packing.append(
                    {
                        "task_id": target["task_id"],
                        "selector": selector,
                        "budget": budget,
                        "target_visible": meta["target_visible"],
                        "target_rank": meta["target_rank"],
                        "n_items": meta["n_items"],
                        "tokens": meta["tokens"],
                    }
                )
    leaks = []
    n_answer = 0
    n_judge = 0
    for target in pilot:
        for condition in config["pilot_conditions"]:
            context, meta = condition_items(target, background, condition)
            leaked = forbidden_in_answer_context(target, context)
            if leaked:
                leaks.append({"task_id": target["task_id"], "condition": condition["id"], "fields": leaked})
            answers = answer_calls(target, context, config)
            judges = judge_calls(target, context, config)
            record = {
                "task_id": target["task_id"],
                "condition": condition,
                "context": context,
                "meta": meta,
                "answers": answers,
                "judges": judges,
            }
            dump_json(
                FRAMEWORK / "requests" / condition["id"] / f"{target['task_id']}.json",
                record,
            )
            n_answer += len(answers)
            n_judge += len(judges)
    if leaks:
        raise RuntimeError(f"answer context leaked label fields: {leaks[:3]}")
    manifest = {
        "selection_rule": (
            "Take the lowest-priority tasks below low_priority_threshold, at most "
            "max_low_priority_in_pilot. Fill the rest with the highest-priority remaining "
            "task in each missing domain, then the highest-priority tasks overall. "
            "The future query is not an input to this rule."
        ),
        "n_tasks_below_threshold": len(failures),
        "task_ids": [item["task_id"] for item in pilot],
        "tasks": [public_task(item) for item in pilot],
    }
    dump_json(FRAMEWORK / "failure_cases.json", {"threshold": threshold, "tasks": failures})
    dump_json(FRAMEWORK / "pilot_manifest.json", manifest)
    packing_path = FRAMEWORK / "packing_all_tasks.jsonl"
    packing_path.parent.mkdir(parents=True, exist_ok=True)
    packing_path.write_text("".join(json.dumps(row) + "\n" for row in packing))
    summary = {
        "n_targets": len(targets),
        "n_background": len(background),
        "n_failure_cases": len(failures),
        "pilot_task_ids": manifest["task_ids"],
        "n_answer_requests": n_answer,
        "n_judge_requests": n_judge,
        "deepseek_called": False,
        "request_dir": str(FRAMEWORK / "requests"),
    }
    dump_json(FRAMEWORK / "build_summary.json", summary)
    return summary


def main() -> None:
    summary = build()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
