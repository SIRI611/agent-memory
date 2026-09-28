"""Selection over write-time units (facts with rules) from memory_rules.

The pool for a task is every unit extracted from the 238 background turns plus
the units extracted from that task's own target turn. A task counts as visible
when at least one selected unit came from the target turn.
"""

from __future__ import annotations

from jev_persist import jev_units as ju
from jev_persist.memory_rules import units_by_turn
from jev_persist.routing import K, corpus, multi_fact_distractors

ACTION_GATE = 0.5


def units_for_turn(text_hash: str) -> list[dict]:
    return [{**u, "turn_hash": text_hash} for u in units_by_turn().get(text_hash, [])]


def unit_pool(task: dict, pool: str = "inmind", stored_target: dict | None = None) -> list[dict]:
    _, background, _ = corpus()
    fact = stored_target or task
    units = [u for b in background for u in units_for_turn(b["text_hash"])]
    units += units_for_turn(fact["text_hash"])
    if pool == "multi_fact":
        units += [u for t in multi_fact_distractors(fact) for u in units_for_turn(t["text_hash"])]
    return units


def necessity(state: str, unit: dict) -> float:
    return ju.cache()[ju.key(ju.FACT_POLICY, state, unit["fact_id"])]


def applicability(state: str, rule: dict) -> float:
    return ju.cache()[ju.key(ju.RULE_POLICY, state, rule["rule_id"])]


def max_applicability(state: str, unit: dict) -> float:
    return max((applicability(state, r) for r in unit["rules"]), default=0.0)


SCORERS = {
    "facts": necessity,
    "rules": lambda s, u: (necessity(s, u) + max_applicability(s, u)) / 2,
    "rules_max": lambda s, u: max(necessity(s, u), max_applicability(s, u)),
    "app_only": max_applicability,
}


def ranked(method: str, state: str, units: list[dict]) -> list[tuple[float, dict]]:
    scorer = SCORERS[method]
    scored = [(scorer(state, u), u) for u in units]
    return sorted(scored, key=lambda row: (-row[0], row[1]["fact_id"]))


def select_units(method: str, state: str, units: list[dict], k: int = K, threshold: float | None = None) -> list[dict]:
    rows = ranked(method, state, units)
    if threshold is not None:
        rows = [row for row in rows if row[0] >= threshold]
    return [u for _, u in rows[:k]]


def render_units(units: list[dict], state: str | None = None, actions: bool = False) -> str:
    lines = []
    for unit in units:
        lines.append(f"- {unit['fact']}")
        if actions and state is not None:
            for rule in unit["rules"]:
                if applicability(state, rule) >= ACTION_GATE:
                    lines.append(f"  - Consider: {rule['then']}")
    return "\n".join(lines)


def visible(task: dict, units: list[dict]) -> bool:
    return any(u["turn_hash"] == task["text_hash"] for u in units)
