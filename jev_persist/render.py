"""Render answer and judge requests. Selection never receives the future query."""

from __future__ import annotations

import json
from pathlib import Path

from jev_persist.corpus import DATA
from jev_persist.pack import pack, target_rank

PROMPTS = DATA / "prompts"
NO_MEMORY = "(no stored memory)"


def load_prompt(name: str) -> str:
    return (PROMPTS / name).read_text()


def context_from_items(items: list[dict]) -> str:
    lines = []
    for item in items:
        text = item.get("text") or item["user_message"]
        lines.append(f"- {text}")
    return "\n".join(lines)


def condition_items(target: dict, background: list[dict], condition: dict) -> tuple[str, dict]:
    selector = condition["selector"]
    budget = condition["budget"]
    if condition["id"] == "oracle":
        context = f"- {target['user_message']}"
        meta = {
            "target_visible": True,
            "target_rank": None,
            "n_items": 1,
            "tokens": target["tokens"],
            "text_hashes": [target["text_hash"]],
        }
        return context, meta
    if condition["id"] == "no_memory":
        return NO_MEMORY, {
            "target_visible": False,
            "target_rank": None,
            "n_items": 0,
            "tokens": 0,
            "text_hashes": [],
        }
    pool = [{"text": target["user_message"], **target}, *background]
    packed = pack(pool, selector, int(budget))
    chosen = packed["chosen"]
    hashes = [item["text_hash"] for item in chosen]
    context = context_from_items(chosen)
    return context, {
        "target_visible": target["text_hash"] in hashes,
        "target_rank": target_rank(packed["ranked"], target["text_hash"]),
        "n_items": len(chosen),
        "tokens": packed["tokens"],
        "text_hashes": hashes,
    }


def answer_request(target: dict, context: str, query: str, config: dict) -> dict:
    system = load_prompt("answer_system.txt").replace("{context}", context)
    return {
        "model": config["answer_model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        "thinking": config["thinking"],
        "max_tokens": config["max_output_tokens"],
    }


def judge_request(prompt_name: str, fields: dict, config: dict) -> dict:
    body = "\n".join(f"{key}: {value}" for key, value in fields.items())
    return {
        "model": config["answer_model"],
        "messages": [
            {"role": "system", "content": load_prompt(prompt_name)},
            {"role": "user", "content": body},
        ],
        "thinking": config["thinking"],
        "max_tokens": config["judge_max_output_tokens"],
        "response_format": {"type": "json_object"},
    }


def answer_calls(target: dict, context: str, config: dict) -> dict[str, dict]:
    return {
        "naive": answer_request(target, context, target["naive_query"], config),
        "query": answer_request(target, context, target["query"], config),
    }


def judge_calls(target: dict, context: str, config: dict) -> dict[str, dict]:
    return {
        "naive": judge_request(
            "judge_naive.txt",
            {
                "user_message": target["user_message"],
                "context": context,
                "query": target["naive_query"],
                "answer": "{answer}",
            },
            config,
        ),
        "application": judge_request(
            "judge_application.txt",
            {
                "user_message": target["user_message"],
                "context": context,
                "query": target["query"],
                "explanation": target["explanation"],
                "answer": "{answer}",
            },
            config,
        ),
        "answer_only": judge_request(
            "judge_answer_only.txt",
            {
                "user_message": target["user_message"],
                "query": target["query"],
                "explanation": target["explanation"],
                "answer": "{answer}",
            },
            config,
        ),
        "target_recall": judge_request(
            "judge_target_recall.txt",
            {
                "user_message": target["user_message"],
                "context": context,
                "query": target["query"],
                "explanation": target["explanation"],
            },
            config,
        ),
    }


def forbidden_in_answer_context(target: dict, context: str) -> list[str]:
    """Strings that would leak the label into the answer prompt."""
    leaked = []
    for name, value in (
        ("explanation", target["explanation"]),
        ("relation", target["relation"]),
        ("naive_query", target["naive_query"]),
        ("query", target["query"]),
    ):
        if value and value in context:
            leaked.append(name)
    return leaked


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
