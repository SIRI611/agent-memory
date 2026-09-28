"""Track DeepSeek spend against a fixed USD budget.

The account endpoint reports remaining balance. This ledger estimates what this
project spent from token counts. A call is refused when the estimate plus a
cache-miss reservation for that call would pass the budget.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from jev_persist.corpus import ROOT, load_config

LEDGER = ROOT / "results" / "framework" / "deepseek_ledger.jsonl"
BALANCE_URL = "https://api.deepseek.com/user/balance"
MILLION = Decimal(1_000_000)
# USD per 1M tokens. Peak is weekday 01:00–04:00 and 06:00–10:00 UTC.
# https://api-docs.deepseek.com/quick_start/pricing
RATES = {
    "flash": {
        "hit": {"off": Decimal("0.003"), "peak": Decimal("0.006")},
        "miss": {"off": Decimal("0.15"), "peak": Decimal("0.30")},
        "out": {"off": Decimal("0.60"), "peak": Decimal("1.20")},
    },
    "pro": {
        "hit": {"off": Decimal("0.022"), "peak": Decimal("0.044")},
        "miss": {"off": Decimal("0.66"), "peak": Decimal("1.32")},
        "out": {"off": Decimal("1.98"), "peak": Decimal("3.96")},
    },
}


def money(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)}"


def is_peak(moment: datetime) -> bool:
    moment = moment.astimezone(timezone.utc)
    if moment.weekday() >= 5:
        return False
    hour = moment.hour
    return hour in {1, 2, 3, 6, 7, 8, 9}


def rate_family(model: str) -> str:
    lowered = model.lower()
    if "pro" in lowered:
        return "pro"
    return "flash"


def estimate_usd(usage: dict, model: str, moment: datetime) -> Decimal:
    hit, miss, completion = split_usage(usage)
    band = "peak" if is_peak(moment) else "off"
    rates = RATES[rate_family(model)]
    return (
        Decimal(hit) * rates["hit"][band]
        + Decimal(miss) * rates["miss"][band]
        + Decimal(completion) * rates["out"][band]
    ) / MILLION


def split_usage(usage: dict) -> tuple[int, int, int]:
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if hit is None and miss is None:
        details = usage.get("prompt_tokens_details") or {}
        hit = int(details.get("cached_tokens") or 0)
        prompt = int(usage.get("prompt_tokens") or 0)
        miss = max(0, prompt - hit)
    else:
        hit = int(hit or 0)
        miss = int(miss or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return hit, miss, completion


def reserve_usage(body: dict) -> dict:
    characters = 0
    for message in body.get("messages") or []:
        characters += len(message.get("content") or "")
    prompt_tokens = max(1, (characters + 3) // 4)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(body.get("max_tokens") or 0),
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": prompt_tokens,
    }


class Budget:
    def __init__(self, limit_usd: Decimal, ledger_path: Path = LEDGER):
        self.limit = limit_usd
        self.ledger_path = ledger_path
        self._stamp: tuple[int, int] | None = None
        self._rows: list[dict] = []
        self._ids: dict[str, dict] = {}
        self._spent = Decimal(0)

    @classmethod
    def from_config(cls) -> "Budget":
        config = load_config()
        return cls(Decimal(str(config["deepseek_budget_usd"])))

    def _refresh(self) -> None:
        """Re-read the ledger only when another writer changed it."""
        if not self.ledger_path.exists():
            self._stamp, self._rows, self._ids, self._spent = None, [], {}, Decimal(0)
            return
        stat = self.ledger_path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        if stamp == self._stamp:
            return
        rows = [json.loads(line) for line in self.ledger_path.read_text().splitlines() if line.strip()]
        self._rows = rows
        self._ids = {row["request_id"]: row for row in rows}
        self._spent = sum((Decimal(row["estimated_usd"]) for row in rows), Decimal(0))
        self._stamp = stamp

    def rows(self) -> list[dict]:
        self._refresh()
        return list(self._rows)

    def spent(self) -> Decimal:
        self._refresh()
        return self._spent

    def remaining(self) -> Decimal:
        return self.limit - self.spent()

    def allow(self, body: dict, moment: datetime | None = None) -> tuple[bool, Decimal]:
        moment = moment or datetime.now(timezone.utc)
        model = str(body.get("model") or "deepseek-flash")
        reservation = estimate_usd(reserve_usage(body), model, moment)
        return self.spent() + reservation <= self.limit, reservation

    def record(
        self,
        *,
        request_id: str,
        model: str,
        usage: dict,
        task_id: int | None = None,
        condition: str | None = None,
        name: str | None = None,
        moment: datetime | None = None,
    ) -> dict:
        moment = moment or datetime.now(timezone.utc)
        self._refresh()
        if request_id in self._ids:
            return self._ids[request_id]
        hit, miss, completion = split_usage(usage)
        cost = estimate_usd(usage, model, moment)
        row = {
            "request_id": request_id,
            "time": moment.isoformat(),
            "peak": is_peak(moment),
            "model": model,
            "task_id": task_id,
            "condition": condition,
            "name": name,
            "cache_hit_tokens": hit,
            "cache_miss_tokens": miss,
            "completion_tokens": completion,
            "estimated_usd": money(cost),
            "spent_after_usd": money(self.spent() + cost),
            "budget_usd": money(self.limit),
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row) + "\n"
        before = self._stamp
        with self.ledger_path.open("a") as handle:
            handle.write(line)
        stat = self.ledger_path.stat()
        if before is not None and stat.st_size == before[0] + len(line.encode()):
            self._rows.append(row)
            self._ids[request_id] = row
            self._spent += cost
            self._stamp = (stat.st_size, stat.st_mtime_ns)
        else:
            self._refresh()
        return row


def fetch_balance(api_key: str) -> dict:
    request = urllib.request.Request(
        BALANCE_URL,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode())


def format_status(budget: Budget, balance: dict | None = None, balance_error: str | None = None) -> str:
    spent = budget.spent()
    lines = [
        f"project budget: ${money(budget.limit)}",
        f"estimated spend: ${money(spent)}",
        f"remaining budget: ${money(budget.remaining())}",
        f"calls: {len(budget.rows())}",
        "estimate uses published Flash/Pro rates and ignores Chinese public holidays",
    ]
    if balance is not None:
        lines.append(f"account available: {balance.get('is_available')}")
        for info in balance.get("balance_infos") or []:
            lines.append(
                f"account {info.get('currency')}: total {info.get('total_balance')} "
                f"(granted {info.get('granted_balance')}, topped up {info.get('topped_up_balance')})"
            )
    if balance_error:
        lines.append(f"account balance: {balance_error}")
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Show DeepSeek spend against the project budget.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also query GET /user/balance. This does not spend tokens.",
    )
    args = parser.parse_args()
    budget = Budget.from_config()
    balance = None
    error = None
    if args.live:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            error = "DEEPSEEK_API_KEY is missing"
        else:
            try:
                balance = fetch_balance(api_key)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
    print(format_status(budget, balance, error))


if __name__ == "__main__":
    main()
