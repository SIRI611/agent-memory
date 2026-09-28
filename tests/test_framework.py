import json
import unittest
from pathlib import Path

from jev_persist.build import build, select_pilot
from jev_persist.corpus import ROOT, load_config, scored_items
from jev_persist.pack import pack
from jev_persist.render import condition_items, forbidden_in_answer_context
from jev_persist.run_deepseek import main as deepseek_main


class FrameworkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = build()
        cls.config = load_config()
        cls.targets, cls.background = scored_items(cls.config)

    def test_build_does_not_claim_a_deepseek_call(self):
        self.assertFalse(self.summary["deepseek_called"])
        self.assertEqual(self.summary["n_answer_requests"], 20 * 7 * 2)

    def test_pilot_contains_low_and_high_priority_tasks(self):
        manifest = json.loads((ROOT / "results/framework/pilot_manifest.json").read_text())
        self.assertEqual(len(manifest["task_ids"]), 20)
        priorities = [item["jev_priority"] for item in manifest["tasks"]]
        self.assertLess(min(priorities), self.config["low_priority_threshold"])
        self.assertGreater(max(priorities), self.config["low_priority_threshold"])

    def test_selection_matches_the_frozen_rule(self):
        chosen = [item["task_id"] for item in select_pilot(self.targets, self.config)]
        manifest = json.loads((ROOT / "results/framework/pilot_manifest.json").read_text())
        self.assertEqual(chosen, manifest["task_ids"])

    def test_answer_requests_do_not_contain_the_bridge(self):
        request_root = ROOT / "results/framework/requests"
        files = list(request_root.glob("*/*.json"))
        self.assertEqual(len(files), 20 * 7)
        for path in files:
            record = json.loads(path.read_text())
            task = next(item for item in self.targets if item["task_id"] == record["task_id"])
            for call in record["answers"].values():
                system = call["messages"][0]["content"]
                user = call["messages"][1]["content"]
                self.assertEqual(forbidden_in_answer_context(task, system), [])
                self.assertNotIn(task["explanation"], system)
                self.assertIn(call["thinking"], ({"type": "disabled"},))
                self.assertEqual(call["model"], "deepseek-flash")
            self.assertEqual(record["answers"]["query"]["messages"][1]["content"], task["query"])
            self.assertEqual(record["answers"]["naive"]["messages"][1]["content"], task["naive_query"])
            if record["condition"]["id"] == "oracle":
                self.assertIn(task["user_message"], record["context"])
            if record["condition"]["id"] == "no_memory":
                self.assertNotIn(task["user_message"], record["context"])
            if record["condition"]["id"].startswith("jev_priority"):
                self.assertLessEqual(record["meta"]["tokens"], record["condition"]["budget"])

    def test_packing_never_exceeds_the_budget(self):
        target = self.targets[0]
        pool = [{"text": target["user_message"], **target}, *self.background]
        for budget in (128, 512):
            packed = pack(pool, "jev_priority", budget)
            self.assertLessEqual(packed["tokens"], budget)

    def test_oracle_context_is_only_the_target(self):
        target = self.targets[0]
        context, meta = condition_items(
            target, self.background, {"id": "oracle", "selector": None, "budget": None}
        )
        self.assertEqual(context, f"- {target['user_message']}")
        self.assertTrue(meta["target_visible"])
        self.assertEqual(
            forbidden_in_answer_context(target, context),
            [],
        )

    def test_dry_run_does_not_call_deepseek(self):
        from io import StringIO
        from unittest.mock import patch

        responses = ROOT / "results/framework/deepseek_responses.jsonl"
        before = responses.read_bytes() if responses.exists() else None
        buffer = StringIO()
        with patch("sys.argv", ["run_deepseek"]), patch("sys.stdout", buffer), \
                patch("jev_persist.run_deepseek.post", side_effect=AssertionError("network call")):
            deepseek_main()
        self.assertIn("dry-run", buffer.getvalue())
        after = responses.read_bytes() if responses.exists() else None
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
