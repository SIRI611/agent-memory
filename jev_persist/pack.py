"""Pack a core-memory list under a token budget. Ties break on text hash."""

from __future__ import annotations


def score_of(item: dict, selector: str) -> float:
    return float(item[selector])


def rank_items(items: list[dict], selector: str) -> list[dict]:
    return sorted(items, key=lambda item: (-score_of(item, selector), item["text_hash"]))


def pack(items: list[dict], selector: str, budget: int) -> dict:
    ranked = rank_items(items, selector)
    chosen = []
    used = 0
    for item in ranked:
        cost = int(item["tokens"])
        if used + cost <= budget:
            chosen.append(item)
            used += cost
    return {"ranked": ranked, "chosen": chosen, "tokens": used}


def target_rank(ranked: list[dict], target_hash: str) -> int:
    for index, item in enumerate(ranked, start=1):
        if item["text_hash"] == target_hash:
            return index
    raise KeyError(target_hash)
