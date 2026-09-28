"""InMind always-in-state baseline (Appendix 17) with a DeepSeek updater.

One markdown memory file is rewritten after every session and truncated to
200 lines and 25,000 bytes. Sessions 1-8 are shared across tasks; the task's
user/assistant pair is appended to session 9; sessions 10-47 follow. The
updater prompt is copied from the paper's appendix. Every step is appended to
results/ais/steps.jsonl so an interrupted run resumes from the last step.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal

from jev_persist.budget import Budget, fetch_balance
from jev_persist.corpus import DATA, ROOT, clean_text, load_config
from jev_persist.routing import corpus
from jev_persist.run_deepseek import post, request_id

OUT = ROOT / "results" / "ais"
STEPS = OUT / "steps.jsonl"
FINAL = OUT / "final"
PHASE_A = 8
MAX_LINES = 200
MAX_BYTES = 25_000
MAX_TOKENS = 8192
SYSTEM = (
    "You maintain a memory file for a personal assistant. Store a fact ONLY if it meets at least one criterion: "
    "1. It would change what advice you give (constraints, risks, needs). "
    "2. The user would be upset or harmed if you forgot it. "
    "Do not store: assistant responses, general knowledge, instructions, opinions on media, idle questions. "
    "One fact per line. Max 200 lines. Output the updated memory file. Nothing else."
)
USER = (
    "## CURRENT MEMORY FILE:\n{memory}\n\n## CONVERSATION:\n{conversation}\n\n"
    "## OUTPUT:\nWrite the complete updated memory file below:"
)


def sessions() -> list[list[dict]]:
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for line in (DATA / "lme_s_background.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            grouped.setdefault(row["id"].split("-turn-")[0], []).append(row)
    return list(grouped.values())


def render_session(turns: list[dict], extra: list[tuple[str, str]] = ()) -> str:
    lines = [f"[{t['role'].upper()}]: {clean_text(t['content'])}" for t in turns]
    lines += [f"[{role}]: {text}" for role, text in extra]
    return "\n".join(lines)


def truncate(memory: str) -> str:
    lines = memory.strip().splitlines()[:MAX_LINES]
    text = "\n".join(lines)
    encoded = text.encode()
    if len(encoded) > MAX_BYTES:
        text = encoded[:MAX_BYTES].decode(errors="ignore")
        text = text.rsplit("\n", 1)[0]
    return text


def body_for(memory: str, conversation: str, config: dict) -> dict:
    return {
        "model": config["answer_model"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(memory=memory, conversation=conversation)},
        ],
        "thinking": config["thinking"],
        "max_tokens": MAX_TOKENS,
    }


class Updater:
    def __init__(self, api_key: str, config: dict):
        self.api_key = api_key
        self.config = config
        self.budget = Budget.from_config()
        self.lock = threading.Lock()
        self.inflight = Decimal(0)
        self.spent = self.budget.spent()
        self.stopped = False
        self.done = {}
        if STEPS.exists():
            for line in STEPS.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.done[(row["track"], row["step"])] = row["memory"]

    def step(self, track: str, step: int, memory: str, conversation: str) -> str | None:
        if (track, step) in self.done:
            return self.done[(track, step)]
        body = body_for(memory, conversation, self.config)
        _, reservation = self.budget.allow(body)
        with self.lock:
            if self.stopped or self.spent + self.inflight + reservation > self.budget.limit:
                self.stopped = True
                return None
            self.inflight += reservation
        content = ""
        try:
            for attempt in range(3):
                payload = post(self.api_key, body, timeout=400)
                content = payload["choices"][0]["message"].get("content") or ""
                key = request_id("ais_update", attempt, track, f"{step}:{uuid.uuid4()}", body)
                with self.lock:
                    recorded = self.budget.record(
                        request_id=key, model=str(payload.get("model") or body["model"]),
                        usage=payload.get("usage") or {}, condition="always_in_state", name=f"{track}:{step}",
                    )
                    self.spent = Decimal(recorded["spent_after_usd"])
                if content.strip():
                    break
        finally:
            with self.lock:
                self.inflight -= reservation
        # An empty completion is an API failure, not an instruction to forget everything.
        updated = truncate(content) if content.strip() else memory
        with self.lock:
            with STEPS.open("a") as handle:
                handle.write(json.dumps({"track": track, "step": step, "memory": updated,
                                         "usage": payload.get("usage"), "estimated_usd": recorded["estimated_usd"]}) + "\n")
            self.done[(track, step)] = updated
        return updated


def run_task(updater: Updater, task: dict, all_sessions: list[list[dict]], prefix: str) -> bool:
    memory = prefix
    track = f"task{task['task_id']}"
    for index in range(PHASE_A, len(all_sessions)):
        extra = ()
        if index == PHASE_A:
            extra = (("USER", task["user_message"]), ("ASSISTANT", task["assistant_message"]))
        memory = updater.step(track, index, memory, render_session(all_sessions[index], extra))
        if memory is None:
            return False
    FINAL.mkdir(parents=True, exist_ok=True)
    (FINAL / f"{task['task_id']}.md").write_text(memory)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    args = parser.parse_args()
    config = load_config()
    all_sessions = sessions()
    targets, _, _ = corpus()
    targets = targets[: args.limit] if args.limit else targets
    n_calls = PHASE_A + len(targets) * (len(all_sessions) - PHASE_A)
    print(f"sessions={len(all_sessions)} tasks={len(targets)} updater_calls={n_calls}", flush=True)
    if not args.execute:
        return
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is missing")
    if not fetch_balance(api_key).get("is_available"):
        raise SystemExit("DeepSeek account is not available")
    OUT.mkdir(parents=True, exist_ok=True)
    updater = Updater(api_key, config)
    memory = ""
    for index in range(PHASE_A):
        memory = updater.step("shared", index, memory, render_session(all_sessions[index]))
        if memory is None:
            raise SystemExit("budget stop during shared prefix")
    print(f"shared prefix done, spent=${updater.spent}", flush=True)
    finished = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_task, updater, t, all_sessions, memory): t for t in targets}
        for future in as_completed(futures):
            task = futures[future]
            try:
                ok = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"task {task['task_id']} failed: {error}", flush=True)
                continue
            finished += ok
            if finished % 10 == 0:
                print(f"finished tasks {finished}/{len(targets)} spent=${updater.spent}", flush=True)
    print(f"done tasks={finished}/{len(targets)} stopped={updater.stopped} spent=${updater.spent}", flush=True)


if __name__ == "__main__":
    main()
