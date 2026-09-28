"""Second-round InMind evaluation: noise repeats, write-time units, LLM scan, query plans.

InMind is now a development set: every task was inspected in round one. The
held-out confirmation runs on other benchmarks.

Conditions (indirect query only; judged with application and answer_only):
- oracle@rN, two_stage_top5@rN: identical request bodies re-sent to measure
  answer plus judge noise. application@jN re-judges round-one answers.
- facts_top5 (P1): top-5 facts by Jev necessity.
- rules_top5 (P2, primary): top-5 facts by mean(necessity, max applicability),
  with matched rule actions shown. rules_top5_noaction: same ranking, facts only.
- oracle_facts / oracle_rules: the target turn's facts, without / with matched actions.
- llm_scan_top5 (P5): DeepSeek reads all stored raw turns and picks at most 5.
- plan_facts_top5 (P4): facts ranked by necessity with query-side cues in the state.
- two_stage_t{tau} (P3): two-stage top-5, keeping only turns with Jev p >= tau.
  Selection rule for the held-out tau, fixed before these answers were scored:
  the largest tau whose application accuracy is within 2 points of the
  three-run two_stage_top5 mean.
Negative controls pair each stored fact with a cross-domain query (round-one pairs).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

from jev_persist import e2e
from jev_persist import units_routing as ur
from jev_persist.budget import Budget, fetch_balance
from jev_persist.corpus import ROOT, load_config
from jev_persist.memory_rules import sha
from jev_persist.offline_report import negative_partners
from jev_persist.render import answer_request, context_from_items, forbidden_in_answer_context, judge_calls, judge_request
from jev_persist.routing import corpus, jev_score, pool_for, select
from jev_persist.run_deepseek import post

OUT = ROOT / "results" / "v2"
ANSWERS = OUT / "answers.jsonl"
JUDGES = OUT / "judgements.jsonl"
SCAN = OUT / "llm_scan.jsonl"
REPEATS = 2
SCAN_SYSTEM = """You are the memory selector of an AI assistant. You receive numbered messages that the user wrote in past conversations, followed by the user's new request.
Select the messages that contain facts about the user that are required to answer the new request safely, correctly, or appropriately (for example, they call for a warning, a restriction, or a different recommendation). Select at most 5, most important first. Select none if none apply.
Return JSON only: {"selected": [numbers]}"""

NOISE = [f"{base}@r{i}" for base in ("oracle", "two_stage_top5") for i in range(1, REPEATS + 1)]
UNIT_CONDITIONS = ["facts_top5", "rules_top5", "rules_top5_noaction", "oracle_facts", "oracle_rules", "plan_facts_top5"]
TAUS = (0.1, 0.2)
THRESHOLDED = [f"two_stage_t{tau}" for tau in TAUS]
CONDITIONS = [*NOISE, *UNIT_CONDITIONS, "llm_scan_top5", *THRESHOLDED, "fuse_top10"]
NEGATIVE = ["neg_facts_top5", "neg_rules_top5", "neg_oracle_rules", "neg_llm_scan_top5", "neg_two_stage",
            *[f"neg_{c}" for c in THRESHOLDED]]
REJUDGE = ["oracle", "two_stage_top5"]


def scan_order(task_id: int, stored_id: int, items: list[dict]) -> list[dict]:
    order = list(items)
    random.Random(f"scan-{task_id}-{stored_id}").shuffle(order)
    return order


def scan_request(query: str, items: list[dict], config: dict) -> dict:
    listing = "\n".join(f"[{i}] {item['text']}" for i, item in enumerate(items, start=1))
    return {
        "model": config["answer_model"],
        "messages": [
            {"role": "system", "content": SCAN_SYSTEM},
            {"role": "user", "content": f"Past messages:\n{listing}\n\nNew request: {query}"},
        ],
        "thinking": config["thinking"],
        "max_tokens": 128,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }


def load_scan() -> dict[str, dict]:
    return e2e.load_rows(SCAN)


def scan_key(query: str, stored: dict) -> str:
    return sha(f"scan\n{query}\n{stored['text_hash']}\n{SCAN_SYSTEM}")


def scan_jobs(config: dict) -> list[dict]:
    targets, _, _ = corpus()
    by_id = {t["task_id"]: t for t in targets}
    pairs = [(t, t) for t in targets] + [(by_id[s], by_id[q]) for s, q in negative_partners().items()]
    jobs = []
    for stored, asked in pairs:
        items = scan_order(asked["task_id"], stored["task_id"], pool_for(stored, stored_target=stored))
        jobs.append({"key": scan_key(asked["query"], stored), "stored": stored["task_id"], "asked": asked["task_id"],
                     "body": scan_request(asked["query"], items, config),
                     "order": [i["text_hash"] for i in items]})
    return jobs


def scan_selected(query: str, stored: dict) -> list[dict] | None:
    row = load_scan_cached().get(scan_key(query, stored))
    if row is None:
        return None
    by_hash = {i["text_hash"]: i for i in pool_for(stored, stored_target=stored)}
    try:
        picked = json.loads(row["content"]).get("selected") or []
    except (json.JSONDecodeError, AttributeError):
        picked = [int(n) for n in re.findall(r"\d+", row["content"] or "")]
    chosen = []
    for n in picked:
        if isinstance(n, int) and 1 <= n <= len(row["order"]):
            item = by_hash[row["order"][n - 1]]
            if item not in chosen:
                chosen.append(item)
    return chosen[:5]


_scan_cache: dict | None = None


def load_scan_cached() -> dict:
    global _scan_cache
    if _scan_cache is None:
        _scan_cache = load_scan()
    return _scan_cache


def unit_context(condition: str, stored: dict, query: str) -> tuple[str, bool] | None:
    try:
        if condition in ("oracle_facts", "oracle_rules"):
            units = ur.units_for_turn(stored["text_hash"])
            return ur.render_units(units, query, actions=condition == "oracle_rules"), True
        units = ur.unit_pool(stored, stored_target=stored)
        if condition == "plan_facts_top5":
            from jev_persist.query_plan import planned_state

            chosen = ur.select_units("facts", planned_state(query), units)
            return ur.render_units(chosen), ur.visible(stored, chosen)
        method = "facts" if condition == "facts_top5" else "rules"
        chosen = ur.select_units(method, query, units)
        return ur.render_units(chosen, query, actions=condition == "rules_top5"), ur.visible(stored, chosen)
    except KeyError:
        return None


def thresholded(stored: dict, query: str, tau: float) -> tuple[str, bool]:
    chosen = [i for i in select("two_stage", query, pool_for(stored, stored_target=stored)) if jev_score(query, i) >= tau]
    return context_from_items(chosen) or e2e.NO_MEMORY, any(i["text_hash"] == stored["text_hash"] for i in chosen)


def context_for(condition: str, stored: dict, query: str) -> tuple[str, bool] | None:
    base = condition.split("@")[0]
    if base in ("oracle", "two_stage_top5"):
        return e2e.context_for(base, stored, "query")
    if base == "two_stage":
        return thresholded(stored, query, 0.0)
    if base.startswith("two_stage_t"):
        return thresholded(stored, query, float(base.removeprefix("two_stage_t")))
    if base == "fuse_top10":
        from jev_persist.probe_inmind import chosen, load

        picked = chosen(stored, load())
        if picked is None:
            return None
        return context_from_items(picked) or e2e.NO_MEMORY, any(i["text_hash"] == stored["text_hash"] for i in picked)
    if base == "llm_scan_top5":
        chosen = scan_selected(query, stored)
        if chosen is None:
            return None
        return context_from_items(chosen) or e2e.NO_MEMORY, any(i["text_hash"] == stored["text_hash"] for i in chosen)
    return unit_context(base, stored, query)


def answer_jobs(config: dict) -> list[dict]:
    targets, _, _ = corpus()
    by_id = {t["task_id"]: t for t in targets}
    jobs = []
    for task in targets:
        for condition in CONDITIONS:
            built = context_for(condition, task, task["query"])
            if built is None:
                continue
            context, visible = built
            if forbidden_in_answer_context(task, context):
                raise RuntimeError(f"label leak in task {task['task_id']} {condition}")
            jobs.append({"kind": "answer", "task_id": task["task_id"], "condition": condition, "name": "query",
                         "body": answer_request(task, context, task["query"], config),
                         "context": context, "target_visible": visible})
    for stored_id, query_id in negative_partners().items():
        stored, asked = by_id[stored_id], by_id[query_id]
        for condition in NEGATIVE:
            built = context_for(condition.removeprefix("neg_"), stored, asked["query"])
            if built is None:
                continue
            context, visible = built
            jobs.append({"kind": "answer", "task_id": stored_id, "query_task": query_id, "condition": condition,
                         "name": "query", "body": answer_request(stored, context, asked["query"], config),
                         "context": context, "target_visible": visible})
    return jobs


def filled_judge(task: dict, context: str, answer: str, judge: str, config: dict) -> dict:
    body = json.loads(json.dumps(judge_calls(task, context, config)[judge]))
    body["messages"][1]["content"] = body["messages"][1]["content"].replace("{answer}", answer or "")
    return body


def judge_jobs(config: dict, answers: dict[str, dict]) -> list[dict]:
    targets, _, _ = corpus()
    by_id = {t["task_id"]: t for t in targets}
    jobs = []
    for row in answers.values():
        task = by_id[row["task_id"]]
        if "query_task" in row:
            asked = by_id[row["query_task"]]
            body = judge_request("judge_overuse.txt", {"user_message": task["user_message"], "query": asked["query"],
                                                       "answer": row["content"] or ""}, config)
            jobs.append({"kind": "judge", "task_id": row["task_id"], "query_task": row["query_task"],
                         "condition": row["condition"], "name": "overuse", "body": body})
            continue
        for judge in ("application", "answer_only"):
            jobs.append({"kind": "judge", "task_id": row["task_id"], "condition": row["condition"], "name": judge,
                         "body": filled_judge(task, row["context"], row["content"], judge, config)})
    old = e2e.load_rows(e2e.ANSWERS)
    for row in old.values():
        if row["condition"] in REJUDGE and row["name"] == "query" and "query_task" not in row:
            task = by_id[row["task_id"]]
            for i in range(1, REPEATS + 1):
                jobs.append({"kind": "judge", "task_id": row["task_id"], "condition": row["condition"],
                             "name": f"application@j{i}",
                             "body": filled_judge(task, row["context"], row["content"], "application", config)})
    return jobs


def run_scan(runner: e2e.Runner, config: dict) -> None:
    done = load_scan()
    jobs = [j for j in scan_jobs(config) if j["key"] not in done]
    print(f"scan jobs pending {len(jobs)}", flush=True)
    wrapped = [{"kind": "scan", "task_id": j["stored"], "condition": "llm_scan", "name": f"asked{j['asked']}",
                "body": j["body"], "order": j["order"], "key": j["key"]} for j in jobs]
    results = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with SCAN.open("a") as handle, ThreadPoolExecutor(runner.workers) as pool:
        futures = {pool.submit(runner.one, job, job["key"]): job for job in wrapped}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"scan failed {job['task_id']}: {error}", flush=True)
                continue
            if result is None:
                continue
            row = {"request_id": job["key"], "stored": job["task_id"], "asked": job["name"], "order": job["order"], **result}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            results[job["key"]] = row
    global _scan_cache
    _scan_cache = None
    print(f"scan done {len(results)} spent=${runner.spent}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--scan", action="store_true", help="run the LLM memory scan stage first")
    parser.add_argument("--only", default="", help="comma-separated condition prefixes to run")
    args = parser.parse_args()
    if args.only:
        keep = tuple(args.only.split(","))
        global CONDITIONS, NEGATIVE
        CONDITIONS = [c for c in CONDITIONS if c.startswith(keep)]
        NEGATIVE = [c for c in NEGATIVE if c.removeprefix("neg_").startswith(keep)]
    config = load_config()
    if args.execute:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit("DEEPSEEK_API_KEY is missing")
        if not fetch_balance(api_key).get("is_available"):
            raise SystemExit("DeepSeek account is not available")
        runner = e2e.Runner(api_key, args.workers)
        if args.scan:
            run_scan(runner, config)
    jobs = answer_jobs(config)
    counts = {}
    for job in jobs:
        counts[job["condition"]] = counts.get(job["condition"], 0) + 1
    print("answer jobs", json.dumps(counts), flush=True)
    if not args.execute:
        return
    answers = e2e.load_rows(ANSWERS)
    runner.run(jobs, ANSWERS, answers, lambda result: {})
    judges = judge_jobs(config, answers)
    judged = e2e.load_rows(JUDGES)
    runner.run(judges, JUDGES, judged, e2e.judge_extra)
    print(f"total spent=${Budget.from_config().spent()}", flush=True)


if __name__ == "__main__":
    main()
