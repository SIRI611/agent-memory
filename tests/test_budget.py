import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from jev_persist.budget import Budget, estimate_usd, is_peak, reserve_usage


class BudgetTest(unittest.TestCase):
    def test_weekday_peak_windows(self):
        self.assertTrue(is_peak(datetime(2026, 9, 23, 2, 0, tzinfo=timezone.utc)))
        self.assertTrue(is_peak(datetime(2026, 9, 23, 8, 30, tzinfo=timezone.utc)))
        self.assertFalse(is_peak(datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc)))
        self.assertFalse(is_peak(datetime(2026, 9, 26, 2, 0, tzinfo=timezone.utc)))

    def test_flash_off_peak_price(self):
        moment = datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc)
        usage = {
            "prompt_cache_hit_tokens": 1_000_000,
            "prompt_cache_miss_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
        }
        self.assertEqual(estimate_usd(usage, "deepseek-flash", moment), Decimal("0.753"))

    def test_missing_cache_fields_count_as_misses(self):
        moment = datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc)
        usage = {"prompt_tokens": 2_000_000, "completion_tokens": 0}
        self.assertEqual(estimate_usd(usage, "deepseek-flash", moment), Decimal("0.30"))

    def test_ledger_stops_at_the_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = Budget(Decimal("100"), Path(directory) / "ledger.jsonl")
            moment = datetime(2026, 9, 23, 2, 0, tzinfo=timezone.utc)
            budget.record(
                request_id="one",
                model="deepseek-flash",
                usage={"prompt_cache_miss_tokens": 0, "completion_tokens": 83_333_000},
                moment=moment,
            )
            body = {
                "model": "deepseek-flash",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "x" * 4000}],
            }
            allowed, _reservation = budget.allow(body, moment)
            self.assertFalse(allowed)
            self.assertLess(budget.spent(), Decimal("100"))
            budget.record(
                request_id="one",
                model="deepseek-flash",
                usage={"completion_tokens": 1},
                moment=moment,
            )
            self.assertEqual(len(budget.rows()), 1)

    def test_reservation_uses_the_output_cap(self):
        usage = reserve_usage(
            {"max_tokens": 1024, "messages": [{"role": "user", "content": "a" * 400}]}
        )
        self.assertEqual(usage["completion_tokens"], 1024)
        self.assertEqual(usage["prompt_cache_hit_tokens"], 0)
        self.assertGreater(usage["prompt_cache_miss_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
