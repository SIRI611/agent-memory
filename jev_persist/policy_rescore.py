"""Rescore cached Jev outputs with two frozen admission rules.

No new Jev calls. DeepSeek is used only for tasks whose target visibility
changes at 512 tokens, and only for the indirect question.
"""

from __future__ import annotations

import json

from jev_persist.corpus import ROOT, load_config, scored_items
from jev_persist.render import answer_request, dump_json, judge_request

OUT = ROOT / "results" / "framework" / "policy_rescore"
BUDGETS = (128, 512)


def user_heavy(item: dict) -> float:
    return 0.70 * float(item["user_specific"]) + 0.30 * float(item["decision_relevant"])


def identity_keep(item: dict) -> float:
    priority = float(item["jev_priority"])
    if item["memory_type"] == "identity" and float(item["user_specific"]) >= 0.8:
        return max(priority, float(item["user_specific"]))
    return priority


RULES = {
    "user_heavy": user_heavy,
    "identity_keep": identity_keep,
}


def annotate(items: list[dict]) -> None:
    for item in items:
        item["user_heavy"] = user_heavy(item)
        item["identity_keep"] = identity_keep(item)


def visible(target: dict, background: list[dict], selector: str, budget: int) -> bool:
    from jev_persist.pack import pack

    pool = [{"text": target["user_message"], **target}, *background]
    chosen = pack(pool, selector, budget)["chosen"]
    return any(item["text_hash"] == target["text_hash"] for item in chosen)


def context_for(target: dict, background: list[dict], selector: str, budget: int) -> str:
    from jev_persist.pack import pack
    from jev_persist.render import context_from_items

    pool = [{"text": target["user_message"], **target}, *background]
    chosen = pack(pool, selector, budget)["chosen"]
    return context_from_items(chosen)


def compare() -> dict:
    config = load_config()
    targets, background = scored_items(config)
    annotate(targets)
    annotate(background)
    base = "jev_priority"
    summary = {"budgets": {}, "rules": list(RULES)}
    flips = []
    for budget in BUDGETS:
        base_vis = {item["task_id"]: visible(item, background, base, budget) for item in targets}
        summary["budgets"][str(budget)] = {
            "jev_priority": sum(base_vis.values()) / len(targets)
        }
        for name in RULES:
            new_vis = {item["task_id"]: visible(item, background, name, budget) for item in targets}
            gained = [tid for tid, flag in new_vis.items() if flag and not base_vis[tid]]
            lost = [tid for tid, flag in new_vis.items() if base_vis[tid] and not flag]
            summary["budgets"][str(budget)][name] = {
                "visible_rate": sum(new_vis.values()) / len(targets),
                "gained": gained,
                "lost": lost,
            }
            if budget == 512:
                for item in targets:
                    if new_vis[item["task_id"]] == base_vis[item["task_id"]]:
                        continue
                    flips.append(
                        {
                            "task": item,
                            "rule": name,
                            "was_visible": base_vis[item["task_id"]],
                            "now_visible": new_vis[item["task_id"]],
                            "context": context_for(item, background, name, budget),
                        }
                    )
    requests = []
    for flip in flips:
        task = flip["task"]
        condition = f"{flip['rule']}_512"
        body = answer_request(task, flip["context"], task["query"], config)
        judge = judge_request(
            "judge_answer_only.txt",
            {
                "user_message": task["user_message"],
                "query": task["query"],
                "explanation": task["explanation"],
                "answer": "{answer}",
            },
            config,
        )
        record = {
            "task_id": task["task_id"],
            "rule": flip["rule"],
            "condition": condition,
            "was_visible": flip["was_visible"],
            "now_visible": flip["now_visible"],
            "user_message": task["user_message"],
            "query": task["query"],
            "answer": body,
            "judge": judge,
        }
        dump_json(OUT / "requests" / condition / f"{task['task_id']}.json", record)
        requests.append(record)
    public = {
        "budgets": summary["budgets"],
        "n_answer_requests": len(requests),
        "note": "Weights were chosen from the 20-task failure analysis, then visibility was measured on all 125. Answers are requested only where 512-token visibility changes.",
    }
    # gained/lost lists make the summary large but useful
    dump_json(OUT / "visibility.json", public)
    return public


def execute() -> None:
    import os

    from jev_persist.budget import Budget, fetch_balance
    from jev_persist.run_deepseek import post, request_id

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is missing")
    budget = Budget.from_config()
    balance = fetch_balance(api_key)
    if not balance.get("is_available"):
        raise SystemExit(f"DeepSeek account is not available: {balance}")
    answers = OUT / "answers.jsonl"
    done = set()
    if answers.exists():
        done = {
            json.loads(line)["request_id"]
            for line in answers.read_text().splitlines()
            if line.strip()
        }
    sent = 0
    with answers.open("a") as handle:
        for path in sorted((OUT / "requests").glob("*/*.json")):
            record = json.loads(path.read_text())
            key = request_id("policy-answer", record["task_id"], record["condition"], "query", record["answer"])
            if key in done:
                continue
            allowed, reservation = budget.allow(record["answer"])
            if not allowed:
                raise SystemExit(f"budget stop, reservation ${reservation}, spent ${budget.spent()}")
            payload = post(api_key, record["answer"])
            usage = payload.get("usage") or {}
            model = str(payload.get("model") or record["answer"]["model"])
            content = payload["choices"][0]["message"].get("content") or ""
            recorded = budget.record(
                request_id=key,
                model=model,
                usage=usage,
                task_id=record["task_id"],
                condition=record["condition"],
                name="query",
            )
            judge_body = json.loads(json.dumps(record["judge"]))
            judge_body["messages"][1]["content"] = judge_body["messages"][1]["content"].replace("{answer}", content)
            judge_key = request_id("policy-judge", record["task_id"], record["condition"], "answer_only", judge_body)
            allowed, reservation = budget.allow(judge_body)
            if not allowed:
                raise SystemExit(f"budget stop before judge, reservation ${reservation}")
            judged = post(api_key, judge_body)
            judge_usage = judged.get("usage") or {}
            judge_content = judged["choices"][0]["message"].get("content") or ""
            budget.record(
                request_id=judge_key,
                model=str(judged.get("model") or judge_body["model"]),
                usage=judge_usage,
                task_id=record["task_id"],
                condition=record["condition"],
                name="answer_only",
            )
            score = None
            try:
                raw = json.loads(judge_content).get("score")
                if raw in {0, 1, "0", "1"}:
                    score = int(raw)
            except json.JSONDecodeError:
                score = None
            row = {
                "request_id": key,
                "task_id": record["task_id"],
                "condition": record["condition"],
                "was_visible": record["was_visible"],
                "now_visible": record["now_visible"],
                "user_message": record["user_message"],
                "score": score,
                "answer": content,
                "estimated_usd": recorded["estimated_usd"],
                "spent_after_usd": recorded["spent_after_usd"],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            sent += 1
            print(
                f"{sent} task={record['task_id']} {record['condition']} "
                f"visible {record['was_visible']}->{record['now_visible']} score={score} "
                f"spent=${budget.spent()}",
                flush=True,
            )
    print(f"new_pairs={sent} spent=${budget.spent()}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    summary = compare()
    print(json.dumps(summary, indent=2)[:4000])
    if args.execute:
        execute()


if __name__ == "__main__":
    main()
