"""Same fusion as IMLogic fuse_top10, on InMind.

Stage 1 is the existing pointwise Jev rank. The probe is the constraint question
from jev_persist.rerank, asked only for the top 100. Selection is
0.7 * necessity + 0.3 * probe, top 10.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from jev_persist import rerank as rr
from jev_persist.routing import corpus, jev_score, pool_for, rank_jev

OUT = rr.h.ROOT / "results" / "v2" / "jev_probe_inmind.jsonl"
_lock = threading.Lock()


def load() -> dict[str, float]:
    out = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["key"]] = row["p"]
    return out


def key(query: str, item: dict) -> str:
    return rr.h.sha(f"inmind-probe\n{rr.PROBE_QUESTION}\n{rr.h.sha(query)}\n{item['text_hash']}")


def run(workers: int, pace: float) -> None:
    from jev_persist import jev_units as ju

    api_key = os.environ["AI_GATEWAY_API_KEY"]
    original = ju.qr._pace
    ju.qr._pace = lambda gap=pace: original(gap)
    done = load()
    targets, _, _ = corpus()
    jobs = []
    for task in targets:
        items = rank_jev(task["query"], pool_for(task))[:100]
        todo = [i for i in items if key(task["query"], i) not in done]
        for start in range(0, len(todo), ju.BATCH):
            jobs.append((task["query"], todo[start:start + ju.BATCH]))
    print(f"inmind probe calls {len(jobs)}", flush=True)

    def one(query: str, batch: list[dict]) -> list[dict]:
        questions = {f"q{n}": {"type": "boolean", "instructions": rr.PROBE_QUESTION.format(text=i["text"])}
                     for n, i in enumerate(batch)}
        payload = ju.post({"model": ju.MODEL, "state": "User request:\n" + query, "questions": questions}, api_key)
        return [{"key": key(query, i), "p": payload["answers"][f"q{n}"]["probability"]} for n, i in enumerate(batch)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    finished = 0
    with OUT.open("a") as handle, ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(one, q, b) for q, b in jobs]):
            try:
                rows = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"failed: {error}", flush=True)
                continue
            with _lock:
                handle.write("".join(json.dumps(r) + "\n" for r in rows))
                handle.flush()
            finished += 1
    print(f"finished {finished}/{len(jobs)}", flush=True)


def chosen(task: dict, cache: dict[str, float], k: int = 10) -> list[dict] | None:
    items = rank_jev(task["query"], pool_for(task))[:100]
    scored = []
    for item in items:
        try:
            probe = cache[key(task["query"], item)]
        except KeyError:
            return None
        need = jev_score(task["query"], item)
        scored.append((0.7 * need + 0.3 * probe, need, item))
    return [item for _, _, item in sorted(scored, key=lambda row: (-row[0], -row[1], row[2]["text_hash"]))][:k]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pace", type=float, default=0.1)
    args = parser.parse_args()
    run(args.workers, args.pace)
