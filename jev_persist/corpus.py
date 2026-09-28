"""Load InMind tasks, background user turns, and cached Jev scores."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "results" / "jev_admission_cache.jsonl"
CONFIG = ROOT / "configs" / "pilot.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text())


def clean_text(text: str) -> str:
    return re.split(r"\n---TIMESTAMP:", text, maxsplit=1)[0].strip()


def text_hash(text: str, config: dict) -> str:
    payload = f"{config['policy_version']}\n{config['jev_model']}\n{text}"
    return hashlib.sha256(payload.encode()).hexdigest()


def token_count(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def load_tasks() -> list[dict]:
    return [
        json.loads(line)
        for line in (DATA / "inmind.jsonl").read_text().splitlines()
        if line.strip()
    ]


def load_background_user_turns() -> list[dict]:
    turns = []
    for line in (DATA / "lme_s_background.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["role"] != "user":
            continue
        turns.append(
            {
                "turn_id": row["id"],
                "text": clean_text(row["content"]),
            }
        )
    return turns


def load_score_cache() -> dict[str, dict]:
    found: dict[str, dict] = {}
    for line in CACHE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("ok"):
            found[row["text_hash"]] = row
    return found


def features(row: dict, config: dict) -> dict[str, float | str]:
    answers = row["answers"]
    type_probs = answers["memory_type"].get("probabilities") or {}
    impact = answers["impact"]
    levels = max(len(impact.get("probabilities") or {}), 1)
    parts: dict[str, float | str] = {
        "constraint": float(type_probs.get("stable_constraint", 0.0)),
        "stable": float(answers["stable"]["probability"]),
        "decision_relevant": float(answers["decision_relevant"]["probability"]),
        "impact": float(impact["score"]) / (levels - 1 if levels > 1 else 1),
        "user_specific": float(answers["user_specific"]["probability"]),
        "memory_type": answers["memory_type"].get("choice") or "",
    }
    weights = config["weights"]
    parts["jev_priority"] = sum(float(weights[name]) * float(parts[name]) for name in weights)
    return parts


def keyword_score(text: str, keywords: list[str]) -> float:
    lowered = text.lower()
    return float(sum(token in lowered for token in keywords))


def random_score(text_hash_value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text_hash_value}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def scored_items(config: dict | None = None) -> tuple[list[dict], list[dict]]:
    config = config or load_config()
    cache = load_score_cache()
    targets = []
    for task in load_tasks():
        text = clean_text(task["user_message"])
        key = text_hash(text, config)
        if key not in cache:
            raise KeyError(f"missing Jev score for task {task['task_id']}")
        parts = features(cache[key], config)
        targets.append(
            {
                "task_id": task["task_id"],
                "domain": task["domain"],
                "user_message": text,
                "assistant_message": task["assistant_message"],
                "naive_query": task["naive_query"],
                "query": task["query"],
                "explanation": task["explanation"],
                "relation": task["relation"],
                "entity_1": task["entity_1"],
                "entity_2": task["entity_2"],
                "text_hash": key,
                "char_len": len(text),
                "tokens": token_count(text),
                "keyword": keyword_score(text, config["keywords"]),
                "shortness": -len(text),
                "random": random_score(key, config["random_seed"]),
                **parts,
            }
        )
    background = []
    seen = set()
    for turn in load_background_user_turns():
        key = text_hash(turn["text"], config)
        if key in seen:
            continue
        seen.add(key)
        if key not in cache:
            raise KeyError(f"missing Jev score for background turn {turn['turn_id']}")
        parts = features(cache[key], config)
        background.append(
            {
                "turn_id": turn["turn_id"],
                "text": turn["text"],
                "text_hash": key,
                "char_len": len(turn["text"]),
                "tokens": token_count(turn["text"]),
                "keyword": keyword_score(turn["text"], config["keywords"]),
                "shortness": -len(turn["text"]),
                "random": random_score(key, config["random_seed"]),
                **parts,
            }
        )
    return targets, background
