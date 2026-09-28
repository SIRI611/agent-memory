"""LongMemEval-S with the same two-stage router as InMind.

A memory is one round: a user turn plus the assistant reply that follows it.
The write-time gate keeps a round when P(user-specific) >= 0.5. Read-time
ranking asks whether that round is required to answer the question, and keeps
the top 5. BM25 and MiniLM rank the same rounds. Abstention items are answered
but left out of recall, matching the paper. The judge prompts are the official
LongMemEval prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from jev_persist.corpus import ROOT, load_config
from jev_persist.heldout_imlogic import ADMIT_QUESTION, bm25

DATA = ROOT / "data" / "heldout" / "longmemeval" / "longmemeval_s_cleaned.json"
OUT = ROOT / "results" / "heldout" / "longmemeval"
JEV = OUT / "jev.jsonl"
ADMIT = OUT / "admit.jsonl"
EMB = OUT / "embed_minilm.json"
ANSWERS = OUT / "answers.jsonl"
JUDGES = OUT / "judgements.jsonl"
K = 5
NECESSITY = (
    "Earlier, the user said: \"{text}\"\n"
    "Is this fact about the user required to answer the request in the state safely, correctly, "
    "or appropriately (for example, it calls for a warning, a restriction, or a different recommendation)?"
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@lru_cache(maxsize=1)
def instances() -> list[dict]:
    return json.loads(DATA.read_text())


def turns(inst: dict) -> list[dict]:
    """One memory is a user turn plus the assistant reply that follows it."""
    out = []
    for session_id, date, session in zip(inst["haystack_session_ids"], inst["haystack_dates"], inst["haystack_sessions"]):
        pending = None
        for index, turn in enumerate(session):
            content = (turn.get("content") or "").strip()
            if turn.get("role") == "user" and content:
                pending = (index, content)
            elif turn.get("role") == "assistant" and pending and content:
                user_index, user = pending
                text = f"User: {user}\nAssistant: {content}"
                out.append({"text": text, "session_id": session_id, "date": date,
                            "hash": sha(text), "uid": f"{session_id}:{user_index}"})
                pending = None
        if pending:
            user_index, user = pending
            text = f"User: {user}"
            out.append({"text": text, "session_id": session_id, "date": date,
                        "hash": sha(text), "uid": f"{session_id}:{user_index}"})
    return out


def state_of(inst: dict) -> str:
    return f"[{inst['question_date']}] {inst['question']}"


def jev_key(state: str, text_hash: str) -> str:
    return sha(f"lme-necessity\n{sha(state)}\n{text_hash}")


def load_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


@lru_cache(maxsize=1)
def jev_cache() -> dict[str, float]:
    return {row["key"]: row["p"] for row in load_jsonl(JEV)}


@lru_cache(maxsize=1)
def admit_cache() -> dict[str, float]:
    return {row["hash"]: row["p"] for row in load_jsonl(ADMIT)}


@lru_cache(maxsize=1)
def embed_cache() -> dict:
    return json.loads(EMB.read_text())


def ranked(method: str, inst: dict) -> list[dict]:
    units = turns(inst)
    if method == "bm25":
        scores = bm25(inst["question"], [u["text"] for u in units])
    elif method == "embed":
        scores = [embed_cache()[inst["question_id"]][u["uid"]] for u in units]
    elif method == "two_stage":
        gate = admit_cache()
        units = [u for u in units if gate.get(u["hash"], 0.0) >= 0.5]
        scores = [jev_cache()[jev_key(state_of(inst), u["hash"])] for u in units]
    else:
        scores = [jev_cache()[jev_key(state_of(inst), u["hash"])] for u in units]
    return [u for _, u in sorted(zip(scores, units), key=lambda row: (-row[0], row[1]["uid"]))]


def recall(method: str) -> dict:
    rows = []
    for inst in instances():
        if inst["question_id"].endswith("_abs"):
            continue
        chosen = ranked(method, inst)[:K]
        got = {u["session_id"] for u in chosen}
        gold = set(inst["answer_session_ids"])
        rows.append((len(got & gold) / len(gold), int(bool(got & gold))))
    n = len(rows)
    return {"n": n, "recall@5": round(sum(r for r, _ in rows) / n, 3), "hit@5": round(sum(h for _, h in rows) / n, 3)}


def run_jev(workers: int, pace: float) -> None:
    from jev_persist import jev_units as ju

    api_key = os.environ["AI_GATEWAY_API_KEY"]
    original = ju.qr._pace
    ju.qr._pace = lambda gap=pace: original(gap)
    done = jev_cache()
    jobs = []
    for inst in instances():
        state = state_of(inst)
        seen = set()
        todo = []
        for unit in turns(inst):
            if unit["hash"] in seen or jev_key(state, unit["hash"]) in done:
                continue
            seen.add(unit["hash"])
            todo.append(unit)
        batch = []
        chars = 0
        for unit in todo:
            if batch and (len(batch) >= 12 or chars + len(unit["text"]) > 24000):
                jobs.append((state, batch))
                batch, chars = [], 0
            batch.append(unit)
            chars += len(unit["text"])
        if batch:
            jobs.append((state, batch))
    print(f"jev calls pending {len(jobs)}", flush=True)
    lock = threading.Lock()

    def one(state: str, batch: list[dict]) -> list[dict]:
        questions = {f"q{i}": {"type": "boolean", "instructions": NECESSITY.format(text=u["text"])} for i, u in enumerate(batch)}
        payload = ju.post({"model": ju.MODEL, "state": "User request:\n" + state, "questions": questions}, api_key)
        return [{"key": jev_key(state, u["hash"]), "p": payload["answers"][f"q{i}"]["probability"]} for i, u in enumerate(batch)]

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
    jev_cache.cache_clear()
    print(f"jev finished {finished}/{len(jobs)}", flush=True)


def run_admit(workers: int, pace: float) -> None:
    from jev_persist import jev_units as ju

    api_key = os.environ["AI_GATEWAY_API_KEY"]
    original = ju.qr._pace
    ju.qr._pace = lambda gap=pace: original(gap)
    done = admit_cache()
    seen = {}
    for inst in instances():
        for unit in turns(inst):
            if unit["hash"] not in done:
                seen.setdefault(unit["hash"], unit["text"])
    units = [{"hash": h, "text": t} for h, t in seen.items()]
    jobs, batch, chars = [], [], 0
    for unit in units:
        if batch and (len(batch) >= 12 or chars + len(unit["text"]) > 24000):
            jobs.append(batch)
            batch, chars = [], 0
        batch.append(unit)
        chars += len(unit["text"])
    if batch:
        jobs.append(batch)
    print(f"admit calls pending {len(jobs)}", flush=True)
    lock = threading.Lock()

    def one(batch: list[dict]) -> list[dict]:
        questions = {f"q{i}": {"type": "boolean", "instructions": f"Memory text:\n{u['text']}\n{ADMIT_QUESTION}"}
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
            if finished % 50 == 0:
                print(f"admit {finished}/{len(jobs)}", flush=True)
    admit_cache.cache_clear()
    print(f"admit finished {finished}/{len(jobs)}", flush=True)


def run_embed() -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    texts, index = {}, {}
    for inst in instances():
        for unit in turns(inst):
            texts.setdefault(unit["hash"], unit["text"])
    hashes = list(texts)
    vectors = model.encode([texts[h] for h in hashes], normalize_embeddings=True, batch_size=128)
    by_hash = {h: vectors[i] for i, h in enumerate(hashes)}
    out = {}
    for inst in instances():
        query = model.encode([inst["question"]], normalize_embeddings=True)[0]
        units = turns(inst)
        out[inst["question_id"]] = {u["uid"]: float(by_hash[u["hash"]] @ query) for u in units}
    OUT.mkdir(parents=True, exist_ok=True)
    EMB.write_text(json.dumps(out))
    print(f"embedded {len(instances())} questions, {len(hashes)} texts")


def context(method: str, inst: dict) -> list[dict] | None:
    if method == "no_memory":
        return []
    if method == "oracle":
        gold = set(inst["answer_session_ids"])
        return [u for u in turns(inst) if u["session_id"] in gold]
    try:
        return ranked(method.removesuffix("_top5"), inst)[:K]
    except KeyError:
        return None


def judge_prompt(inst: dict, hypothesis: str) -> str:
    task, question, answer = inst["question_type"], inst["question"], inst["answer"]
    abstention = inst["question_id"].endswith("_abs")
    if abstention:
        return ("I will give you an unanswerable question, an explanation, and a response from a model. "
                "Please answer yes if the model correctly identifies the question as unanswerable. The model could "
                "say that the information is incomplete, or some other information is given but the asked information is not.\n\n"
                f"Question: {question}\n\nExplanation: {answer}\n\nModel Response: {hypothesis}\n\n"
                "Does the model correctly identify the question as unanswerable? Answer yes or no only.")
    shared = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
              "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct "
              "answer or contains all the intermediate steps to get the correct answer, you should also answer yes. "
              "If the response only contains a subset of the information required by the answer, answer no.")
    if task == "temporal-reasoning":
        shared += (" In addition, do not penalize off-by-one errors for the number of days. If the question asks for "
                   "the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 "
                   "days when the answer is 18), the model's response is still correct.")
    elif task == "knowledge-update":
        shared += (" If the response contains some previous information along with an updated answer, the response "
                   "should be considered as correct as long as the updated answer is the required answer.")
    elif task == "single-session-preference":
        return ("I will give you a question, a rubric for desired personalized response, and a response from a model. "
                "Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does "
                "not need to reflect all the points in the rubric. The response is correct as long as it recalls and "
                "utilizes the user's personal information correctly.\n\n"
                f"Question: {question}\n\nRubric: {answer}\n\nModel Response: {hypothesis}\n\n"
                "Is the model response correct? Answer yes or no only.")
    return f"{shared}\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {hypothesis}\n\nIs the model response correct? Answer yes or no only."


def answer_body(inst: dict, units: list[dict], config: dict) -> dict:
    memory = "\n".join(f"- [{u['date']}] {u['text']}" for u in units) or "(none)"
    prompt = (f"Question date: {inst['question_date']}\nQuestion: {inst['question']}\n\n"
              f"What the user said before:\n{memory}\n\n"
              "Answer the question from these memories. If they do not contain the answer, say you don't know.")
    return {"model": config["answer_model"], "messages": [{"role": "user", "content": prompt}],
            "thinking": config["thinking"], "temperature": 0, "max_tokens": 256}


def run_answers(methods: list[str], workers: int) -> None:
    from jev_persist import e2e

    config = load_config()
    runner = e2e.Runner(os.environ["DEEPSEEK_API_KEY"], workers)
    jobs = []
    for inst in instances():
        for method in methods:
            units = context(method, inst)
            if units is None:
                continue
            gold = set(inst["answer_session_ids"])
            jobs.append({"kind": "answer", "task_id": inst["question_id"], "condition": f"lme:{method}", "name": "qa",
                         "body": answer_body(inst, units, config),
                         "hit": any(u["session_id"] in gold for u in units)})
    print(f"answer jobs {len(jobs)}", flush=True)
    runner.run(jobs, ANSWERS, e2e.load_rows(ANSWERS), lambda result: {})
    answers = e2e.load_rows(ANSWERS)
    judges = []
    by_id = {inst["question_id"]: inst for inst in instances()}
    for row in answers.values():
        if not row["condition"].startswith("lme:"):
            continue
        inst = by_id[row["task_id"]]
        if f"lme:{row['condition'].split(':',1)[1]}" != row["condition"]:
            continue
        method = row["condition"].split(":", 1)[1]
        if method not in methods:
            continue
        judges.append({"kind": "judge", "task_id": inst["question_id"], "condition": row["condition"], "name": "yesno",
                       "body": {"model": config["answer_model"], "messages": [{"role": "user", "content": judge_prompt(inst, row["content"] or "")}],
                                "thinking": config["thinking"], "temperature": 0, "max_tokens": 8}})
    runner.run(judges, JUDGES, e2e.load_rows(JUDGES), lambda result: {})


def report(methods: list[str]) -> dict:
    from jev_persist import e2e

    judged = {}
    for row in e2e.load_rows(JUDGES).values():
        if row["condition"].startswith("lme:"):
            label = "yes" in (row["content"] or "").lower()
            judged.setdefault(row["condition"].split(":", 1)[1], {})[row["task_id"]] = int(label)
    by_id = {inst["question_id"]: inst for inst in instances()}
    out = {"retrieval": {}, "accuracy": {}}
    for method in methods:
        if method.endswith("_top5"):
            try:
                out["retrieval"][method] = recall(method.removesuffix("_top5"))
            except (KeyError, FileNotFoundError):
                pass
        labels = judged.get(method)
        if not labels:
            continue
        ids = list(labels)
        correct = [labels[i] for i in ids]
        by_type = {}
        for i in ids:
            by_type.setdefault(by_id[i]["question_type"], []).append(labels[i])
        out["accuracy"][method] = {
            "n": len(correct), "acc": round(sum(correct) / len(correct), 3),
            "by_type": {t: {"n": len(v), "acc": round(sum(v) / len(v), 3)} for t, v in sorted(by_type.items())},
        }
    (OUT / "report.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jev", action="store_true")
    parser.add_argument("--admit", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--answer", default="")
    parser.add_argument("--retrieval", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pace", type=float, default=0.1)
    args = parser.parse_args()
    if args.embed:
        run_embed()
    if args.admit:
        run_admit(args.workers, args.pace)
    if args.jev:
        run_jev(args.workers, args.pace)
    if args.retrieval:
        for method in args.retrieval.split(","):
            print(method, recall(method), flush=True)
    if args.answer:
        methods = args.answer.split(",")
        run_answers(methods, args.workers)
        print(json.dumps(report(methods), indent=2))


if __name__ == "__main__":
    main()
