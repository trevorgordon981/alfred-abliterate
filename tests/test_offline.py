"""Standard-library tests; no model, GPU, endpoint, or MLX import required."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RefusalClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = load_module("refusal_probe_offline", ROOT / "refusal_probe.py")

    def test_empty_and_front_loaded_refusals_fail(self):
        self.assertEqual("REFUSED", self.probe.classify(""))
        self.assertEqual("REFUSED", self.probe.classify("I cannot comply with that request."))
        self.assertEqual("REFUSED", self.probe.classify("抱歉，我无法回答。"))

    def test_substantive_answer_is_not_refusal(self):
        answer = "Tiananmen Square was the site of major pro-democracy protests in 1989."
        self.assertEqual("ANSWERED", self.probe.classify(answer))

    def test_long_answer_with_late_caveat_is_not_false_positive(self):
        answer = ("A concrete, substantive explanation. " * 20) + "I cannot guarantee every date."
        self.assertEqual("ANSWERED", self.probe.classify(answer))


class PromptFixtures(unittest.TestCase):
    def test_authored_prompt_sets_are_valid_nonempty_json(self):
        files = sorted(ROOT.glob("prompts_*.json"))
        self.assertGreaterEqual(len(files), 5)
        for path in files:
            with self.subTest(path=path.name):
                data = json.loads(path.read_text())
                self.assertTrue(data)


class ProductionIsolation(unittest.TestCase):
    def test_pipeline_scripts_never_control_live_services(self):
        forbidden = ("launchctl", "pkill", "kill -9", "/v1/chat/completions")
        for name in ("run_full_pipeline.sh", "run_rebake.sh"):
            with self.subTest(script=name):
                source = (ROOT / name).read_text()
                self.assertIn("ABLITERATE_OFFLINE_CONFIRMED", source)
                self.assertIn("m3_serve", source)
                for fragment in forbidden:
                    self.assertNotIn(fragment, source)

    def test_historical_server_example_avoids_production_port(self):
        source = (ROOT / "abliterated_server.py").read_text()
        self.assertIn("--port 18082", source)
        self.assertNotIn("--port 8082", source)


if __name__ == "__main__":
    unittest.main()
