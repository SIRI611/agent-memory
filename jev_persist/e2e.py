"""End-to-end InMind evaluation on all 125 tasks with one DeepSeek answer model.

Default is a dry run that counts requests. --execute calls DeepSeek under the
shared $100 ledger. Answers that already exist for an identical request body
(from the pilot) are reused, not re-sent.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal

from jev_persist.budget import Budget, fetch_balance
from jev_persist.corpus import ROOT
from jev_persist.offline_report import negative_partners
from jev_persist.render import (
    NO_MEMORY,
    answer_request,
    context_from_items,
    forbidden_in_answer_context,
    judge_calls,
    judge_request,
)
from jev_persist.routing import corpus, pool_for, select
from jev_persist.run_deepseek import post, request_id

OUT = ROOT / "results" / "e2e"
ANSWERS = OUT / "answers.jsonl"
JUDGES = OUT / "judgements.jsonl"
PILOT_ANSWERS = ROOT / "results" / "framework" / "deepseek_cache.jsonl"
PILOT_JUDGES = ROOT / "results" / "framework" / "deepseek_judgements.jsonl"
AIS_DIR = ROOT / "results" / "ais" / "final"

TOPK_METHODS = {
    "bm25_top5": "bm25",
    "embed_top5": "embed",
    "jev_top5": "jev",
    "two_stage_top5": "two_stage",
    "jevmem_style_top5": "jevmem_style",
}
CONDITIONS = ["no_memory", "oracle", "jev_priority_512", *TOPK_METHODS, "always_in_state"]
NEGATIVE_CONDITIONS = ["neg_oracle", "neg_jev_top5"]


def ais_memory(task_id: int) -> str | None:
    path = AIS_DIR / f"{task_id}.md"
    return path.read_text() if path.exists() else None


def context_for(condition: str, task: dict, query_field: str) -> tuple[str, bool] | None:
    if condition == "no_memory":
        return NO_MEMORY, False
    if condition == "oracle":
        return f"- {task['user_message']}", True
    if condition == "always_in_state":
        memory = ais_memory(task["task_id"])
        if memory is None:
            return None
        return memory.strip() or NO_MEMORY, task["user_message"] in memory
    items = pool_for(task)
    if condition == "jev_priority_512":
        chosen = select("write_v01_512", task[query_field], items)
    else:
        try:
            chosen = select(TOPK_METHODS[condition], task[query_field], items)
        except KeyError:
            return None
    visible = any(i["text_hash"] == task["text_hash"] for i in chosen)
    return context_from_items(chosen), visible


def answer_jobs(config: dict) -> list[dict]:
    targets, _, _ = corpus()
    jobs = []
    for task in targets:
        for condition in CONDITIONS:
            for name, field in (("naive", "naive_query"), ("query", "query")):
                built = context_for(condition, task, field)
                if built is None:
                    continue
                context, visible = built
                if condition != "always_in_state" and forbidden_in_answer_context(task, context):
                    raise RuntimeError(f"label leak in task {task['task_id']} {condition}")
                body = answer_request(task, context, task[field], config)
                jobs.append(
                    {
                        "kind": "answer",
                        "task_id": task["task_id"],
                        "condition": condition,
                        "name": name,
                        "body": body,
                        "context": context,
                        "target_visible": visible,
                    }
                )
    return jobs


def negative_answer_jobs(config: dict) -> list[dict]:
    targets, _, _ = corpus()
    by_id = {t["task_id"]: t for t in targets}
    jobs = []
    for stored_id, query_id in negative_partners().items():
        stored, asked = by_id[stored_id], by_id[query_id]
        for condition in NEGATIVE_CONDITIONS:
            if condition == "neg_oracle":
                context = f"- {stored['user_message']}"
            else:
                chosen = select("jev", asked["query"], pool_for(stored, stored_target=stored))
                context = context_from_items(chosen)
            jobs.append(
                {
                    "kind": "answer",
                    "task_id": stored_id,
                    "query_task": query_id,
                    "condition": condition,
                    "name": "query",
                    "body": answer_request(stored, context, asked["query"], config),
                    "context": context,
                    "target_visible": stored["user_message"] in context,
                }
            )
    return jobs


def judge_jobs(config: dict, answers: dict[str, dict]) -> list[dict]:
    targets, _, _ = corpus()
    by_id = {t["task_id"]: t for t in targets}
    index = {(r["task_id"], r["condition"], r["name"]): r for r in answers.values() if "query_task" not in r}
    jobs = []
    for (task_id, condition, name), row in index.items():
        if name != "query":
            continue
        task = by_id[task_id]
        naive = index.get((task_id, condition, "naive"))
        if naive is None:
            continue
        bodies = judge_calls(task, row["context"], config)
        naive_bodies = judge_calls(task, naive["context"], config)
        fill = {
            "naive": (naive_bodies["naive"], naive["content"]),
            "application": (bodies["application"], row["content"]),
            "answer_only": (bodies["answer_only"], row["content"]),
        }
        for judge_name, (body, answer) in fill.items():
            payload = json.loads(json.dumps(body))
            payload["messages"][1]["content"] = payload["messages"][1]["content"].replace("{answer}", answer or "")
            jobs.append({"kind": "judge", "task_id": task_id, "condition": condition, "name": judge_name, "body": payload})
    for row in answers.values():
        if "query_task" not in row:
            continue
        stored, asked = by_id[row["task_id"]], by_id[row["query_task"]]
        body = judge_request(
            "judge_overuse.txt",
            {"user_message": stored["user_message"], "query": asked["query"], "answer": row["content"] or ""},
            config,
        )
        jobs.append(
            {
                "kind": "judge",
                "task_id": row["task_id"],
                "query_task": row["query_task"],
                "condition": row["condition"],
                "name": "overuse",
                "body": body,
            }
        )
    return jobs


def job_key(job: dict) -> str:
    name = job["name"] if "query_task" not in job else f"{job['name']}@{job['query_task']}"
    return request_id(job["kind"], job["task_id"], job["condition"], name, job["body"])


def load_rows(*paths) -> dict[str, dict]:
    rows = {}
    for path in paths:
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows[row["request_id"]] = row
    return rows


class Runner:
    def __init__(self, api_key: str, workers: int):
        self.api_key = api_key
        self.workers = workers
        self.budget = Budget.from_config()
        self.lock = threading.Lock()
        self.inflight = Decimal(0)
        self.spent = self.budget.spent()
        self.stopped = False

    def reserve(self, body: dict) -> Decimal | None:
        _, reservation = self.budget.allow(body)
        with self.lock:
            if self.stopped or self.spent + self.inflight + reservation > self.budget.limit:
                self.stopped = True
                return None
            self.inflight += reservation
            return reservation

    def one(self, job: dict, key: str) -> dict | None:
        reservation = self.reserve(job["body"])
        if reservation is None:
            return None
        try:
            payload = post(self.api_key, job["body"])
        finally:
            with self.lock:
                self.inflight -= reservation
        usage = payload.get("usage") or {}
        model = str(payload.get("model") or job["body"]["model"])
        content = payload["choices"][0]["message"].get("content") or ""
        with self.lock:
            recorded = self.budget.record(
                request_id=key, model=model, usage=usage, task_id=job["task_id"],
                condition=job["condition"], name=job["name"],
            )
            self.spent = Decimal(recorded["spent_after_usd"])
        return {"content": content, "usage": usage, "estimated_usd": recorded["estimated_usd"]}

    def run(self, jobs: list[dict], out_path, done: dict[str, dict], extra) -> int:
        pending = [(job, job_key(job)) for job in jobs]
        pending = [(job, key) for job, key in pending if key not in done]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sent = 0
        with out_path.open("a") as handle, ThreadPoolExecutor(self.workers) as pool:
            futures = {pool.submit(self.one, job, key): (job, key) for job, key in pending}
            for future in as_completed(futures):
                job, key = futures[future]
                try:
                    result = future.result()
                except Exception as error:  # noqa: BLE001
                    print(f"failed {job['task_id']} {job['condition']} {job['name']}: {error}", flush=True)
                    continue
                if result is None:
                    continue
                row = {"request_id": key, **{k: v for k, v in job.items() if k != "body"}, **result, **extra(result)}
                with self.lock:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                done[key] = row
                sent += 1
                if sent % 100 == 0:
                    print(f"{out_path.name}: {sent}/{len(pending)} spent=${self.spent}", flush=True)
        print(f"{out_path.name}: sent={sent} pending={len(pending)} stopped={self.stopped} spent=${self.spent}", flush=True)
        return sent


def parse_json(content: str) -> dict:
    try:
        value = json.loads(content)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def judge_extra(result: dict) -> dict:
    parsed = parse_json(result["content"])
    out = {}
    for field in ("score", "relevant", "mentioned"):
        if parsed.get(field) in {0, 1, "0", "1"}:
            out[field] = int(parsed[field])
    return out


def reuse_pilot(jobs: list[dict], done: dict[str, dict], pilot: dict[str, dict], path) -> int:
    """Copy pilot rows whose request body is identical into the e2e file."""
    added = 0
    with path.open("a") as handle:
        for job in jobs:
            if "query_task" in job:
                continue
            key = request_id(job["kind"], job["task_id"], job["condition"], job["name"], job["body"])
            if key in done or key not in pilot:
                continue
            source = pilot[key]
            row = {"request_id": key, **{k: v for k, v in job.items() if k != "body"}, "content": source.get("content"),
                   "usage": source.get("usage"), "estimated_usd": "0", "reused_from_pilot": True}
            if job["kind"] == "judge":
                row.update(judge_extra({"content": source.get("content") or ""}))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            done[key] = row
            added += 1
    return added


def main() -> None:
    from jev_persist.corpus import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = load_config()
    jobs = answer_jobs(config) + negative_answer_jobs(config)
    answers = load_rows(ANSWERS)
    counts = {}
    for job in jobs:
        counts[job["condition"]] = counts.get(job["condition"], 0) + 1
    print("answer jobs", json.dumps(counts))
    if not args.execute:
        print(f"dry run: {len(jobs)} answer jobs, {len(answers)} already on disk")
        return
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is missing")
    if not fetch_balance(api_key).get("is_available"):
        raise SystemExit("DeepSeek account is not available")
    OUT.mkdir(parents=True, exist_ok=True)
    reused = reuse_pilot(jobs, answers, load_rows(PILOT_ANSWERS), ANSWERS)
    print(f"reused pilot answers: {reused}", flush=True)
    runner = Runner(api_key, args.workers)
    runner.run(jobs, ANSWERS, answers, lambda result: {})
    judges = judge_jobs(config, answers)
    judged = load_rows(JUDGES)
    reused = reuse_pilot(judges, judged, load_rows(PILOT_JUDGES), JUDGES)
    print(f"judge jobs {len(judges)}, reused pilot judgements: {reused}", flush=True)
    runner.run(judges, JUDGES, judged, judge_extra)


if __name__ == "__main__":
    main()
