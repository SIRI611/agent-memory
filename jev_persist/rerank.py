"""Contrastive second-pass reranking on IMLogic.

Stage 1 is the existing pointwise Jev fact-necessity score over the user's whole
memory bank. Stage 2 rescores the stage-1 top-M with the candidates visible:
- jev_contrast: one Jev call per item, state = request + the M candidates,
  one boolean question per candidate asking whether it is among the few most
  decisive, non-redundant, non-superseded facts.
- llm_rerank: DeepSeek reads the same listing and selects at most K.
Both arms see the same listing, shuffled with seed "rerank-{id}-{M}" so that
list position does not carry the stage-1 order. Ties in jev_contrast break
by the stage-1 score.

Controls: jev_top{K} (stage 1 only) and jev_top10 / jev_top20 (more context,
no reranking).

Result on dev: both rerank arms raised target recall@5 but pulled in the trap
memory as well, and accuracy did not improve (see results/rerank/report_dev.json).

Trap-aware variant (dev exploration, designed after seeing the rerank result):
two extra pointwise questions with state = request only, asked for the stage-1
top-M. trap_constraint = need * P(hard constraint rather than a topical wish);
trap_mislead = need * (1 - P(following it alone could still be wrong)).
Result on dev: trap_constraint cut trap-without-target from 14% to 4-7% but
raised Trap_Generic picks; no variant beat jev_top10 (74%).

Dev split = the 100 IMLogic items already scored in heldout_imlogic. M is
chosen on dev from MS by target recall@5 (ties -> smaller M) before the
held-out split is sampled.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from jev_persist import heldout_imlogic as h

OUT = h.ROOT / "results" / "rerank"
JEV = OUT / "jev_contrast.jsonl"
LLM = OUT / "llm_rerank.jsonl"
ANSWERS = OUT / "answers.jsonl"
K = 5
MS = (20, 30, 50)
POLICY = "v0.4-contrast"
CONTRAST_QUESTION = (
    "Candidate [{n}]: \"{text}\"\n"
    "Compared with the other candidates listed in the state, is this fact one of the few most decisive for "
    "answering the request correctly (knowing it changes what the right answer should be), and not redundant "
    "with or superseded by a more decisive or more recent candidate?"
)
LLM_SYSTEM = """You are the memory selector of an AI assistant. You receive numbered facts about the user from memory (with timestamps), followed by the user's new request.
Select the facts that are most decisive for answering the new request correctly: facts that change what the right answer should be (for example, they call for a warning, a restriction, or a different recommendation). Skip facts that are redundant with, or superseded by, a more decisive or more recent fact. Select at most 5, most important first. Select none if none apply.
Return JSON only: {"selected": [numbers]}"""
TRAP = OUT / "jev_trap.jsonl"
CHOICE = OUT / "jev_choice.jsonl"
HYPE = OUT / "hype.jsonl"
PROBE = OUT / "jev_probe.jsonl"
PROBE_M = 100
PROBE_QUESTION = (
    "Stored fact about the user: \"{text}\"\n"
    "The request can look answerable from a preference or goal that matches its wording. "
    "Is this fact a separate constraint, such as money, health, law, obligations, or a newer value, "
    "that would make that obvious answer unsafe or wrong?"
)
HYPE_M = 50
HYPE_SYSTEM = """You write the request a user would send if this stored fact were the one fact that changes the correct answer.
The request must not quote the fact and must not reuse its distinctive names, numbers, or keywords.
Return JSON only: {"request": "<one sentence>"}"""
CHOICE_M = 30
CHOICE_INSTRUCTIONS = (
    "Which stored fact is required to answer the request in the state correctly "
    "(knowing it changes what the right answer should be)? "
    "A fact that only matches the topic, or that repeats another fact, is not the one."
)
TRAP_QUESTIONS = {
    "v0.4-constraint": (
        "Stored fact about the user: \"{text}\"\n"
        "Is this fact a hard constraint on the user's situation (for example money, health, legal status, "
        "obligations, schedule, or relationships) that limits which answers to the request in the state are "
        "acceptable, rather than a wish, taste, or goal that merely matches the topic of the request?"
    ),
    "v0.4-mislead": (
        "Stored fact about the user: \"{text}\"\n"
        "If an assistant answered the request in the state by following this fact, could the answer still be "
        "wrong for the user because other circumstances (such as money, health, obligations, or recent changes) "
        "matter more?"
    ),
}
_lock = threading.Lock()


def stage1(item: dict) -> list[dict]:
    return h.ranked("jev", item)


def stage1_score(item: dict, unit: dict) -> float:
    return h.jev_cache()[h.jev_key(h.state_of(item), unit["hash"])]


def listing(item: dict, m: int) -> list[dict]:
    order = stage1(item)[:m]
    random.Random(f"rerank-{item['id']}-{m}").shuffle(order)
    return order


def render(units: list[dict]) -> str:
    return "\n".join(f"[{n}] ({u['timestamp']}) {u['memory_content']}" for n, u in enumerate(units, start=1))


def contrast_state(item: dict, m: int) -> str:
    return f"User request:\n{h.state_of(item)}\n\nCandidate facts about the user (from memory):\n{render(listing(item, m))}"


def contrast_key(item: dict, m: int, unit: dict) -> str:
    return h.sha(f"{POLICY}\n{h.sha(contrast_state(item, m))}\n{unit['hash']}")


def load_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


@lru_cache(maxsize=1)
def contrast_cache() -> dict[str, float]:
    return {r["key"]: r["p"] for r in load_jsonl(JEV)}


def run_contrast(items: list[dict], ms: tuple[int, ...], workers: int, pace: float) -> None:
    from jev_persist import jev_units as ju

    api_key = os.environ["AI_GATEWAY_API_KEY"]
    original = ju.qr._pace
    ju.qr._pace = lambda gap=pace: original(gap)
    done = contrast_cache()
    jobs = [(item, m) for item in items for m in ms
            if any(contrast_key(item, m, u) not in done for u in listing(item, m))]
    print(f"contrast calls pending {len(jobs)}", flush=True)

    def one(item: dict, m: int) -> list[dict]:
        units = listing(item, m)
        questions = {f"q{n}": {"type": "boolean", "instructions": CONTRAST_QUESTION.format(n=n, text=u["memory_content"])}
                     for n, u in enumerate(units, start=1)}
        payload = ju.post({"model": ju.MODEL, "state": contrast_state(item, m), "questions": questions}, api_key)
        return [{"key": contrast_key(item, m, u), "id": item["id"], "m": m, "p": payload["answers"][f"q{n}"]["probability"]}
                for n, u in enumerate(units, start=1)]

    OUT.mkdir(parents=True, exist_ok=True)
    finished = 0
    with JEV.open("a") as handle, ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(one, i, m) for i, m in jobs]):
            try:
                rows = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"failed: {error}", flush=True)
                continue
            with _lock:
                handle.write("".join(json.dumps(r) + "\n" for r in rows))
                handle.flush()
            finished += 1
    contrast_cache.cache_clear()
    print(f"contrast finished {finished}/{len(jobs)}", flush=True)


def trap_key(policy: str, item: dict, unit: dict) -> str:
    return h.sha(f"{policy}\n{h.sha(h.state_of(item))}\n{unit['hash']}")


@lru_cache(maxsize=1)
def trap_cache() -> dict[str, float]:
    return {r["key"]: r["p"] for r in load_jsonl(TRAP)}


def run_trap(items: list[dict], m: int, workers: int, pace: float) -> None:
    from jev_persist import jev_units as ju

    api_key = os.environ["AI_GATEWAY_API_KEY"]
    original = ju.qr._pace
    ju.qr._pace = lambda gap=pace: original(gap)
    done = trap_cache()
    jobs = []
    for item in items:
        todo = [(p, u) for p in TRAP_QUESTIONS for u in stage1(item)[:m] if trap_key(p, item, u) not in done]
        for start in range(0, len(todo), ju.BATCH):
            jobs.append((item, todo[start:start + ju.BATCH]))
    print(f"trap calls pending {len(jobs)}", flush=True)

    def one(item: dict, batch: list[tuple[str, dict]]) -> list[dict]:
        questions = {f"q{n}": {"type": "boolean", "instructions": TRAP_QUESTIONS[p].format(text=u["memory_content"])}
                     for n, (p, u) in enumerate(batch)}
        payload = ju.post({"model": ju.MODEL, "state": "User request:\n" + h.state_of(item), "questions": questions}, api_key)
        return [{"key": trap_key(p, item, u), "id": item["id"], "policy": p, "p": payload["answers"][f"q{n}"]["probability"]}
                for n, (p, u) in enumerate(batch)]

    OUT.mkdir(parents=True, exist_ok=True)
    finished = 0
    with TRAP.open("a") as handle, ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(one, i, b) for i, b in jobs]):
            try:
                rows = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"failed: {error}", flush=True)
                continue
            with _lock:
                handle.write("".join(json.dumps(r) + "\n" for r in rows))
                handle.flush()
            finished += 1
    trap_cache.cache_clear()
    print(f"trap finished {finished}/{len(jobs)}", flush=True)


def trap_select(variant: str, item: dict, m: int, k: int) -> list[dict] | None:
    cache = trap_cache()
    scored = []
    for u in stage1(item)[:m]:
        need = stage1_score(item, u)
        try:
            if variant == "constraint":
                s = need * cache[trap_key("v0.4-constraint", item, u)]
            elif variant == "mislead":
                s = need * (1 - cache[trap_key("v0.4-mislead", item, u)])
            else:
                raise ValueError(variant)
        except KeyError:
            return None
        scored.append((s, need, u))
    return [u for _, _, u in sorted(scored, key=lambda r: (-r[0], -r[1], r[2]["hash"]))][:k]


def choice_key(item: dict, m: int) -> str:
    ids = ",".join(u["hash"] for u in listing(item, m))
    return h.sha(f"choice\n{CHOICE_INSTRUCTIONS}\n{h.sha(h.state_of(item))}\n{ids}")


@lru_cache(maxsize=1)
def choice_cache() -> dict[str, dict]:
    return {r["key"]: r for r in load_jsonl(CHOICE)}


def run_choice(items: list[dict], m: int, workers: int, pace: float) -> None:
    from jev_persist import jev_units as ju

    api_key = os.environ["AI_GATEWAY_API_KEY"]
    original = ju.qr._pace
    ju.qr._pace = lambda gap=pace: original(gap)
    done = choice_cache()
    jobs = [item for item in items if choice_key(item, m) not in done]
    print(f"choice calls pending {len(jobs)}", flush=True)

    def one(item: dict) -> dict:
        units = listing(item, m)
        criteria = {f"c{n}": u["memory_content"] for n, u in enumerate(units)}
        criteria["none"] = "None of these facts would change the correct answer."
        payload = ju.post({
            "model": ju.MODEL,
            "state": "User request:\n" + h.state_of(item),
            "questions": {"pick": {"type": "choice", "instructions": CHOICE_INSTRUCTIONS, "criteria": criteria}},
        }, api_key)
        probs = payload["answers"]["pick"].get("probabilities") or {}
        return {"key": choice_key(item, m), "id": item["id"], "m": m,
                "probs": {f"c{n}": float(probs.get(f"c{n}") or 0) for n in range(len(units))},
                "none": float(probs.get("none") or 0),
                "choice": payload["answers"]["pick"].get("choice")}

    OUT.mkdir(parents=True, exist_ok=True)
    finished = 0
    with CHOICE.open("a") as handle, ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(one, i) for i in jobs]):
            try:
                row = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"failed: {error}", flush=True)
                continue
            with _lock:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            finished += 1
    choice_cache.cache_clear()
    print(f"choice finished {finished}/{len(jobs)}", flush=True)


def probe_key(item: dict, unit: dict) -> str:
    return h.sha(f"probe\n{PROBE_QUESTION}\n{h.sha(h.state_of(item))}\n{unit['hash']}")


@lru_cache(maxsize=1)
def probe_cache() -> dict[str, float]:
    return {r["key"]: r["p"] for r in load_jsonl(PROBE)}


def run_probe(items: list[dict], m: int, workers: int, pace: float) -> None:
    from jev_persist import jev_units as ju

    api_key = os.environ["AI_GATEWAY_API_KEY"]
    original = ju.qr._pace
    ju.qr._pace = lambda gap=pace: original(gap)
    done = probe_cache()
    jobs = []
    for item in items:
        todo = [u for u in stage1(item)[:m] if probe_key(item, u) not in done]
        for start in range(0, len(todo), ju.BATCH):
            jobs.append((item, todo[start:start + ju.BATCH]))
    print(f"probe calls pending {len(jobs)}", flush=True)

    def one(item: dict, batch: list[dict]) -> list[dict]:
        questions = {f"q{n}": {"type": "boolean", "instructions": PROBE_QUESTION.format(text=u["memory_content"])}
                     for n, u in enumerate(batch)}
        payload = ju.post({"model": ju.MODEL, "state": "User request:\n" + h.state_of(item), "questions": questions}, api_key)
        return [{"key": probe_key(item, u), "id": item["id"], "p": payload["answers"][f"q{n}"]["probability"]}
                for n, u in enumerate(batch)]

    OUT.mkdir(parents=True, exist_ok=True)
    finished = 0
    with PROBE.open("a") as handle, ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(one, i, b) for i, b in jobs]):
            try:
                rows = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"failed: {error}", flush=True)
                continue
            with _lock:
                handle.write("".join(json.dumps(row) + "\n" for row in rows))
                handle.flush()
            finished += 1
    probe_cache.cache_clear()
    print(f"probe finished {finished}/{len(jobs)}", flush=True)


def rank_probe(item: dict, k: int, alpha: float) -> list[dict] | None:
    """alpha=1 ranks the stage-1 top 100 by the constraint probe alone."""
    cache = probe_cache()
    units = stage1(item)[:PROBE_M]
    scored = []
    for unit in units:
        try:
            probe = cache[probe_key(item, unit)]
        except KeyError:
            return None
        need = stage1_score(item, unit)
        scored.append(((1 - alpha) * need + alpha * probe, need, unit))
    return [unit for _, _, unit in sorted(scored, key=lambda row: (-row[0], -row[1], row[2]["hash"]))][:k]


def probe_select(item: dict, k: int) -> list[dict] | None:
    """Top-k by necessity, then up to 3 more from ranks 11-50 with constraint probability >= 0.3."""
    cache = trap_cache()
    base = stage1(item)[:k]
    have = {u["hash"] for u in base}
    extra = []
    for u in stage1(item)[k:50]:
        try:
            c = cache[trap_key("v0.4-constraint", item, u)]
        except KeyError:
            return None
        if c >= 0.3:
            extra.append((c, stage1_score(item, u), u))
    extra = [u for _, _, u in sorted(extra, key=lambda row: (-row[0], -row[1], row[2]["hash"])) if u["hash"] not in have][:3]
    return base + extra


def hype_jobs(items: list[dict], m: int) -> list[dict]:
    from jev_persist.corpus import load_config

    config = load_config()
    seen = {}
    for item in items:
        for unit in stage1(item)[:m]:
            seen.setdefault(unit["hash"], unit["memory_content"])
    jobs = []
    for digest, text in seen.items():
        body = {"model": config["answer_model"], "messages": [
            {"role": "system", "content": HYPE_SYSTEM},
            {"role": "user", "content": f"Stored fact:\n{text}"},
        ], "thinking": config["thinking"], "max_tokens": 80, "temperature": 0,
                "response_format": {"type": "json_object"}}
        jobs.append({"kind": "hype", "task_id": digest[:16], "condition": "hype", "name": "request",
                     "body": body, "hash": digest})
    return jobs


def run_hype(items: list[dict], m: int, workers: int) -> None:
    from jev_persist import e2e

    jobs = hype_jobs(items, m)
    done = {r.get("hash") for r in load_jsonl(HYPE)}
    pending = [j for j in jobs if j["hash"] not in done]
    print(f"hype pending {len(pending)} of {len(jobs)}", flush=True)
    runner = e2e.Runner(os.environ["DEEPSEEK_API_KEY"], workers)
    runner.run(pending, HYPE, e2e.load_rows(HYPE), lambda result: {})


@lru_cache(maxsize=1)
def hype_requests() -> dict[str, str]:
    out = {}
    for row in load_jsonl(HYPE):
        try:
            text = json.loads(row["content"]).get("request") or ""
        except (json.JSONDecodeError, AttributeError, KeyError):
            text = ""
        if row.get("hash") and text:
            out[row["hash"]] = text
    return out


def choice_select(item: dict, m: int, k: int) -> list[dict] | None:
    row = choice_cache().get(choice_key(item, m))
    if row is None:
        return None
    units = listing(item, m)
    scored = [(row["probs"].get(f"c{n}", 0.0), stage1_score(item, u), u) for n, u in enumerate(units)]
    return [u for _, _, u in sorted(scored, key=lambda r: (-r[0], -r[1], r[2]["hash"]))][:k]


def llm_body(item: dict, m: int, config: dict) -> dict:
    return {
        "model": config["answer_model"],
        "messages": [{"role": "system", "content": LLM_SYSTEM},
                     {"role": "user", "content": f"Facts about the user:\n{render(listing(item, m))}\n\nNew request: {h.state_of(item)}"}],
        "thinking": config["thinking"], "max_tokens": 128, "temperature": 0, "response_format": {"type": "json_object"},
    }


def llm_job(item: dict, m: int, config: dict) -> dict:
    return {"kind": "rerank", "task_id": item["id"], "condition": f"llm_rerank_m{m}", "name": "select",
            "body": llm_body(item, m, config)}


def run_llm(items: list[dict], ms: tuple[int, ...], workers: int) -> None:
    from jev_persist import e2e
    from jev_persist.corpus import load_config

    config = load_config()
    runner = e2e.Runner(os.environ["DEEPSEEK_API_KEY"], workers)
    runner.run([llm_job(i, m, config) for i in items for m in ms], LLM, e2e.load_rows(LLM), lambda result: {})


@lru_cache(maxsize=1)
def llm_rows() -> dict[str, dict]:
    from jev_persist import e2e

    return e2e.load_rows(LLM)


def select(method: str, item: dict) -> list[dict] | None:
    """Units put in context, or None when a needed score is missing."""
    if method.startswith("jev_top"):
        return stage1(item)[:int(method.removeprefix("jev_top"))]
    if method == "fill_cons":
        return probe_select(item, 10)
    if method == "probe_top10":
        return rank_probe(item, 10, 1.0)
    if method == "fuse_top10":
        return rank_probe(item, 10, 0.3)
    if method.startswith("choice_"):
        _, m, k = method.split("_")
        return choice_select(item, int(m.removeprefix("m")), int(k.removeprefix("k")))
    if method.startswith("trap_"):
        _, variant, m, k = method.split("_")
        return trap_select(variant, item, int(m.removeprefix("m")), int(k.removeprefix("k")))
    m = int(method.rsplit("_m", 1)[1])
    units = listing(item, m)
    if method.startswith("jev_contrast"):
        cache = contrast_cache()
        try:
            scored = [(cache[contrast_key(item, m, u)], stage1_score(item, u), u) for u in units]
        except KeyError:
            return None
        return [u for _, _, u in sorted(scored, key=lambda r: (-r[0], -r[1], r[2]["hash"]))][:K]
    if method.startswith("llm_rerank"):
        from jev_persist import e2e
        from jev_persist.corpus import load_config

        row = llm_rows().get(e2e.job_key(llm_job(item, m, load_config())))
        if row is None:
            return None
        try:
            picked = json.loads(row["content"]).get("selected") or []
        except (json.JSONDecodeError, AttributeError):
            picked = [int(n) for n in re.findall(r"\d+", row["content"] or "")]
        chosen = []
        for n in picked:
            if isinstance(n, int) and 1 <= n <= len(units) and units[n - 1] not in chosen:
                chosen.append(units[n - 1])
        return chosen[:K]
    raise ValueError(method)


def methods(ms: tuple[int, ...]) -> list[str]:
    return ["jev_top5", "jev_top10", "jev_top20", "choice_m30_k5", "choice_m30_k10",
            *[f"{arm}_m{m}" for m in ms for arm in ("jev_contrast", "llm_rerank")],
            *[f"trap_{v}_m{m}_k{k}" for v in ("constraint", "mislead") for m in ms for k in (5, 10)]]


def offline(items: list[dict], ms: tuple[int, ...]) -> dict:
    out = {}
    for method in methods(ms):
        chosen = [(i, select(method, i)) for i in items]
        chosen = [(i, c) for i, c in chosen if c is not None]
        if not chosen:
            continue
        n = len(chosen)
        out[method] = {
            "n": n,
            "recall": round(sum(any(u["memory_content"] == i["memory_l"] for u in c) for i, c in chosen) / n, 3),
            "trap_in_context": round(sum(any(u["memory_content"] == i["memory_s"] for u in c) for i, c in chosen) / n, 3),
            "mean_units": round(sum(len(c) for _, c in chosen) / n, 2),
        }
    return out


def answer_jobs(items: list[dict], method_list: list[str]) -> list[dict]:
    from jev_persist.corpus import load_config

    config = load_config()
    jobs = []
    for item in items:
        for method in method_list:
            units = select(method, item)
            if units is None:
                continue
            body, letters = h.answer_body(item, units, config)
            jobs.append({"kind": "answer", "task_id": item["id"], "condition": f"rerank:{method}", "name": "mcq",
                         "body": body, "letters": letters,
                         "target_visible": any(u["memory_content"] == item["memory_l"] for u in units),
                         "s_visible": any(u["memory_content"] == item["memory_s"] for u in units)})
    return jobs


def run_answers(items: list[dict], method_list: list[str], workers: int) -> None:
    from jev_persist import e2e

    runner = e2e.Runner(os.environ["DEEPSEEK_API_KEY"], workers)
    jobs = answer_jobs(items, method_list)
    print(f"answer jobs {len(jobs)}", flush=True)
    runner.run(jobs, ANSWERS, e2e.load_rows(ANSWERS), lambda result: {})


def correctness(ids: set[str], method: str) -> dict[str, int]:
    from jev_persist import e2e

    out = {}
    for row in e2e.load_rows(ANSWERS).values():
        if row["condition"] != f"rerank:{method}" or row["task_id"] not in ids:
            continue
        match = re.search(r"\b([A-D])\b", (row["content"] or "").strip().upper())
        out[row["task_id"]] = int(bool(match) and row["letters"].get(match.group(1), {}).get("option_key") == "Correct")
    return out


def paired(a: dict[str, int], b: dict[str, int]) -> dict:
    ids = sorted(set(a) & set(b))
    diffs = [a[i] - b[i] for i in ids]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(5000))
    return {"n": len(ids), "diff": round(sum(diffs) / len(diffs), 3), "ci95": [round(boots[125], 3), round(boots[4874], 3)],
            "a_better": sum(d > 0 for d in diffs), "b_better": sum(d < 0 for d in diffs)}


def report(items: list[dict], method_list: list[str], name: str) -> dict:
    ids = {i["id"] for i in items}
    acc = {m: correctness(ids, m) for m in method_list}
    out = {"offline": offline(items, MS), "accuracy": {}, "paired": {}}
    for m, c in acc.items():
        if c:
            out["accuracy"][m] = {"n": len(c), "acc": round(sum(c.values()) / len(c), 3)}
    for m in method_list:
        for base in ("jev_top5", "jev_top20"):
            if m != base and acc.get(m) and acc.get(base):
                out["paired"][f"{m} - {base}"] = paired(acc[m], acc[base])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"report_{name}.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev"], default="dev")
    parser.add_argument("--contrast", action="store_true")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--trap", action="store_true")
    parser.add_argument("--choice", action="store_true")
    parser.add_argument("--hype", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--answer", default="", help="comma-separated methods to answer")
    parser.add_argument("--ms", default=",".join(map(str, MS)))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pace", type=float, default=0.1)
    args = parser.parse_args()
    items = h.sample()
    ms = tuple(int(m) for m in args.ms.split(","))
    if args.contrast:
        run_contrast(items, ms, args.workers, args.pace)
    if args.llm:
        run_llm(items, ms, args.workers)
    if args.trap:
        run_trap(items, max(ms), args.workers, args.pace)
    if args.choice:
        run_choice(items, CHOICE_M, args.workers, args.pace)
    if args.hype:
        run_hype(items, HYPE_M, args.workers)
    if args.probe:
        run_probe(items, PROBE_M, args.workers, args.pace)
    if args.answer:
        run_answers(items, args.answer.split(","), args.workers)
    print(json.dumps(report(items, args.answer.split(",") if args.answer else [], args.split), indent=1))


if __name__ == "__main__":
    main()
