"""Held-out confirmation on IMLogic (RootMem, arXiv 2606.23283).

Frozen before any IMLogic score was computed:
- Sample: 5 QA items per user (20 users), random.Random("imlogic-heldout-v1").
- Pool: the user's whole HaluMem memory bank (exact duplicate strings merged).
  Target = the unit equal to memory_l; memory_s is the built-in hard negative.
- Retrievers, all top-5: bm25, embed (all-MiniLM-L6-v2), jev (v0.3 fact
  necessity question, state = "[query_time] query").
- Answer: the official IMLogic multiple-choice prompt and context format,
  option shuffle seed 42, DeepSeek temperature 0. Accuracy = picks "Correct".
- Controls: no_memory, oracle (memory_l only), oracle_plus_s (memory_l and memory_s).

Exploratory, added after the InMind dev threshold sweep (the pre-stated tau
rule selected no threshold): jev_t{tau} = jev top-5 keeping p >= tau.

Stages: --embed (run with .venv-embed python), --jev, --answer. No stage sends
memory_l, memory_s, options, or category to a retriever.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "heldout" / "imlogic"
OUT = ROOT / "results" / "heldout" / "imlogic"
EMBED = OUT / "embed_minilm.json"
JEV = OUT / "jev.jsonl"
ADMIT = OUT / "admit.jsonl"
ADMIT_QUESTION = (
    "Is this a persistent fact about the user, their environment, obligations, preferences, or constraints, "
    "rather than a general question or small talk?"
)
ANSWERS = OUT / "answers.jsonl"
PER_USER = 5
SEED = "imlogic-heldout-v1"
SHUFFLE_SEED = 42
K = 5
RETRIEVERS = ["bm25", "embed", "jev", "two_stage"]
EXPLORATORY = ["jev_t0.1", "jev_t0.2"]
CONDITIONS = ["no_memory", "oracle", "oracle_plus_s", *[f"{r}_top5" for r in RETRIEVERS], *EXPLORATORY]

sys.path.insert(0, str(DATA / "eval"))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@lru_cache(maxsize=1)
def dialogues() -> list[dict]:
    return [json.loads(line) for line in (DATA / "dialogue" / "HaluMem-Medium.jsonl").read_text().splitlines() if line.strip()]


@lru_cache(maxsize=None)
def bank(user: int) -> list[dict]:
    seen = {}
    for session in dialogues()[user]["sessions"]:
        for memory in session.get("memory_points", []):
            content = memory.get("memory_content", "")
            if content and content not in seen:
                seen[content] = {"memory_content": content, "memory_type": memory.get("memory_type", ""),
                                 "timestamp": memory.get("timestamp"), "hash": sha(content)}
    return list(seen.values())


@lru_cache(maxsize=1)
def sample() -> list[dict]:
    rng = random.Random(SEED)
    items = []
    for user in range(20):
        qa = json.loads((DATA / "qa" / f"qa_user{user:02d}.json").read_text())
        for index in sorted(rng.sample(range(len(qa)), PER_USER)):
            items.append({"id": f"u{user:02d}q{index:03d}", "user": user, "index": index, **qa[index]})
    return items


def state_of(item: dict) -> str:
    from evaluate_common import build_prompt_query

    return build_prompt_query(item)


def tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def bm25(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    import math

    toks = [tokenize(d) for d in docs]
    avg = sum(len(t) for t in toks) / max(1, len(toks))
    df = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    n = len(docs)
    scores = []
    for t in toks:
        tf = {}
        for w in t:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for w in set(tokenize(query)):
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avg))
        scores.append(s)
    return scores


@lru_cache(maxsize=1)
def embed_sims() -> dict:
    return json.loads(EMBED.read_text()) if EMBED.exists() else {}


@lru_cache(maxsize=1)
def jev_cache() -> dict:
    out = {}
    if JEV.exists():
        for line in JEV.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["key"]] = row["p"]
    return out


def jev_key(state: str, memory_hash: str) -> str:
    from jev_persist import jev_units as ju

    return ju.key(ju.FACT_POLICY, state, memory_hash)


@lru_cache(maxsize=1)
def admit_cache() -> dict:
    out = {}
    if ADMIT.exists():
        for line in ADMIT.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["hash"]] = row["p"]
    return out


def ranked(retriever: str, item: dict) -> list[dict]:
    units = bank(item["user"])
    state = state_of(item)
    if retriever == "bm25":
        scores = bm25(item["query"], [u["memory_content"] for u in units])
    elif retriever == "embed":
        sims = embed_sims()[item["id"]]
        scores = [sims[u["hash"]] for u in units]
    else:
        cache = jev_cache()
        scores = [cache[jev_key(state, u["hash"])] for u in units]
        if retriever == "two_stage":
            kept = [u for u in units if admit_cache().get(u["hash"], 0.0) >= 0.5]
            scores = [cache[jev_key(state, u["hash"])] for u in kept]
            units = kept
    return [u for _, u in sorted(zip(scores, units), key=lambda row: (-row[0], row[1]["hash"]))]


def context_units(condition: str, item: dict) -> list[dict] | None:
    units = {u["memory_content"]: u for u in bank(item["user"])}
    if condition == "no_memory":
        return []
    if condition == "oracle":
        return [units[item["memory_l"]]]
    if condition == "oracle_plus_s":
        return [units[item["memory_l"]], units[item["memory_s"]]]
    try:
        if condition in EXPLORATORY:
            tau, state = float(condition.removeprefix("jev_t")), state_of(item)
            return [u for u in ranked("jev", item)[:K] if jev_cache()[jev_key(state, u["hash"])] >= tau]
        return ranked(condition.removesuffix("_top5"), item)[:K]
    except KeyError:
        return None


def run_embed() -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    out = {}
    for user in range(20):
        units = bank(user)
        vectors = model.encode([u["memory_content"] for u in units], normalize_embeddings=True, batch_size=128)
        for item in [i for i in sample() if i["user"] == user]:
            q = model.encode([item["query"]], normalize_embeddings=True)[0]
            out[item["id"]] = {u["hash"]: float(v @ q) for u, v in zip(units, vectors)}
    OUT.mkdir(parents=True, exist_ok=True)
    EMBED.write_text(json.dumps(out))
    print(f"embedded {len(out)} queries")


def run_admit(workers: int, pace: float) -> None:
    """Write-time gate, once per memory. Same question as the InMind two-stage gate."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from jev_persist import jev_units as ju

    api_key = os.environ["AI_GATEWAY_API_KEY"]
    original = ju.qr._pace
    ju.qr._pace = lambda gap=pace: original(gap)
    done = admit_cache()
    units = []
    seen = set()
    for user in range(20):
        for unit in bank(user):
            if unit["hash"] not in done and unit["hash"] not in seen:
                seen.add(unit["hash"])
                units.append(unit)
    jobs = [units[start:start + ju.BATCH] for start in range(0, len(units), ju.BATCH)]
    print(f"admit calls pending {len(jobs)}", flush=True)
    lock = threading.Lock()

    def one(batch: list[dict]) -> list[dict]:
        questions = {f"q{i}": {"type": "boolean", "instructions": f"Memory text:\n{u['memory_content']}\n{ADMIT_QUESTION}"}
                     for i, u in enumerate(batch)}
        payload = ju.post({"model": ju.MODEL, "state": "A stored memory is quoted in each question.", "questions": questions}, api_key)
        return [{"hash": u["hash"], "p": payload["answers"][f"q{i}"]["probability"]} for i, u in enumerate(batch)]

    OUT.mkdir(parents=True, exist_ok=True)
    finished = 0
    with ADMIT.open("a") as handle, ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(one, b) for b in jobs]):
            try:
                rows = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"failed: {error}", flush=True)
                continue
            with lock:
                handle.write("".join(json.dumps(r) + "\n" for r in rows))
                handle.flush()
            finished += 1
    admit_cache.cache_clear()
    print(f"admit finished {finished}/{len(jobs)}", flush=True)


def run_jev(workers: int, pace: float) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from jev_persist import jev_units as ju

    api_key = os.environ["AI_GATEWAY_API_KEY"]
    original = ju.qr._pace
    ju.qr._pace = lambda gap=pace: original(gap)
    done = jev_cache()
    jobs = []
    for item in sample():
        state = state_of(item)
        todo = [u for u in bank(item["user"]) if jev_key(state, u["hash"]) not in done]
        for start in range(0, len(todo), ju.BATCH):
            jobs.append((state, todo[start:start + ju.BATCH]))
    print(f"jev calls pending {len(jobs)}", flush=True)
    lock = threading.Lock()

    def one(state: str, batch: list[dict]) -> list[dict]:
        questions = {f"q{i}": {"type": "boolean", "instructions": ju.FACT_QUESTION.format(text=u["memory_content"])}
                     for i, u in enumerate(batch)}
        payload = ju.post({"model": ju.MODEL, "state": "User request:\n" + state, "questions": questions}, api_key)
        return [{"key": jev_key(state, u["hash"]), "p": payload["answers"][f"q{i}"]["probability"]}
                for i, u in enumerate(batch)]

    OUT.mkdir(parents=True, exist_ok=True)
    finished = 0
    with JEV.open("a") as handle, ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(one, s, b) for s, b in jobs]):
            try:
                rows = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"failed: {error}", flush=True)
                continue
            with lock:
                handle.write("".join(json.dumps(r) + "\n" for r in rows))
                handle.flush()
            finished += 1
            if finished % 50 == 0:
                print(f"jev {finished}/{len(jobs)}", flush=True)
    print(f"jev finished {finished}/{len(jobs)}", flush=True)


def answer_body(item: dict, units: list[dict], config: dict) -> tuple[dict, dict]:
    from evaluate_common import (
        DEFAULT_SYSTEM_PROMPT, MULTIPLE_CHOICE_PROMPT_TEMPLATE, build_global_state, build_lettered_options,
        format_context,
    )

    options_str, letters = build_lettered_options(item, item["user"], item["index"], SHUFFLE_SEED)
    dialogue = dialogues()[item["user"]]
    full_bank = [m for s in dialogue["sessions"] for m in s.get("memory_points", [])]
    prompt = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
        global_state=build_global_state(dialogue, full_bank, None), context=format_context(units),
        query=state_of(item), options_str=options_str,
    )
    body = {"model": config["answer_model"], "messages": [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                                                          {"role": "user", "content": prompt}],
            "thinking": config["thinking"], "max_tokens": 16, "temperature": 0}
    return body, letters


def run_answer(workers: int) -> None:
    from jev_persist import e2e
    from jev_persist.corpus import load_config

    config = load_config()
    runner = e2e.Runner(os.environ["DEEPSEEK_API_KEY"], workers)
    jobs = []
    for item in sample():
        for condition in CONDITIONS:
            units = context_units(condition, item)
            if units is None:
                continue
            body, letters = answer_body(item, units, config)
            jobs.append({"kind": "answer", "task_id": item["id"], "condition": f"imlogic:{condition}", "name": "mcq",
                         "body": body, "letters": letters,
                         "target_visible": any(u["memory_content"] == item["memory_l"] for u in units),
                         "s_visible": any(u["memory_content"] == item["memory_s"] for u in units)})
    print(f"answer jobs {len(jobs)}", flush=True)
    runner.run(jobs, ANSWERS, e2e.load_rows(ANSWERS), lambda result: {})


def report() -> dict:
    import re

    from jev_persist import e2e
    from jev_persist.e2e_report import ci

    rows = [r for r in e2e.load_rows(ANSWERS).values()]
    out = {}
    for condition in CONDITIONS:
        picked = []
        for r in [r for r in rows if r["condition"] == f"imlogic:{condition}"]:
            text = (r["content"] or "").strip().upper()
            match = re.search(r"\b([A-D])\b", text)
            key = r["letters"].get(match.group(1), {}).get("option_key") if match else None
            picked.append((r, key))
        if not picked:
            continue
        correct = [int(k == "Correct") for _, k in picked]
        out[condition] = {
            "n": len(picked), "acc": round(sum(correct) / len(correct), 3), "ci95": ci(correct),
            "target_in_context": round(sum(r["target_visible"] for r, _ in picked) / len(picked), 3),
            "s_in_context": round(sum(r["s_visible"] for r, _ in picked) / len(picked), 3),
            "picks": {k: sum(1 for _, x in picked if x == k) for k in
                      ("Correct", "Trap_Preference", "Trap_Fabrication", "Trap_Generic", None)},
        }
    retrieval = {}
    for retriever in RETRIEVERS:
        try:
            ranks = []
            for item in sample():
                order = [u["memory_content"] for u in ranked(retriever, item)]
                def place(text: str) -> int:
                    return order.index(text) + 1 if text in order else len(order) + 1
                ranks.append((place(item["memory_l"]), place(item["memory_s"])))
        except KeyError:
            continue
        n = len(ranks)
        retrieval[retriever] = {
            "recall@1": round(sum(r <= 1 for r, _ in ranks) / n, 3),
            "recall@5": round(sum(r <= 5 for r, _ in ranks) / n, 3),
            "recall@10": round(sum(r <= 10 for r, _ in ranks) / n, 3),
            "s_above_l": round(sum(s < r for r, s in ranks) / n, 3),
            "median_rank_l": sorted(r for r, _ in ranks)[n // 2],
        }
    result = {"answers": out, "retrieval": retrieval, "n_items": len(sample()),
              "mean_pool": round(sum(len(bank(u)) for u in range(20)) / 20, 1)}
    (OUT / "report.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--jev", action="store_true")
    parser.add_argument("--admit", action="store_true")
    parser.add_argument("--answer", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pace", type=float, default=0.1)
    args = parser.parse_args()
    if args.embed:
        run_embed()
    if args.jev:
        run_jev(args.workers, args.pace)
    if args.admit:
        run_admit(args.workers, args.pace)
    if args.answer:
        run_answer(args.workers)
    print(json.dumps(report(), indent=2))


if __name__ == "__main__":
    main()
