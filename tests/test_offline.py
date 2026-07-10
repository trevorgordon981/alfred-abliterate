"""Standard-library tests; no model, GPU, endpoint, or MLX import required."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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


class M3QuantizationOrder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = ROOT / "bake_abliteration_m3.py"
        # The helper must be importable without numpy, MLX, or a model. Numeric
        # dependencies are lazy and are reached only by the bound fused builder.
        cls.helper = load_module("m3_full_precision_helper", cls.script)

    def _model(self, config):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "config.json").write_text(json.dumps(config))
        self.addCleanup(temporary.cleanup)
        return root

    def _direct_command(self, model, *extra):
        return [
            sys.executable,
            "-I",
            str(self.script),
            "--model",
            str(model),
            "--directions",
            "does-not-exist.npz",
            "--output",
            str(Path(model).parent / "must-not-exist"),
            *extra,
        ]

    def test_quantized_config_fails_before_optional_imports_or_direction_reads(self):
        model = self._model({"quantization": {"bits": 6, "group_size": 64}})
        result = subprocess.run(
            self._direct_command(model), capture_output=True, text=True, check=False,
        )
        self.assertEqual(2, result.returncode)
        self.assertIn("REFUSING", result.stderr)
        self.assertIn("low-bit M3 weights must never be edited", result.stderr)
        self.assertIn("fused_abliterate_quantize.py", result.stderr)

    def test_even_full_precision_direct_execution_routes_to_fused_builder(self):
        model = self._model({"text_config": {"num_hidden_layers": 60}})
        result = subprocess.run(
            self._direct_command(model), capture_output=True, text=True, check=False,
        )
        self.assertEqual(2, result.returncode)
        self.assertIn("standalone M3 baking is disabled", result.stderr)
        self.assertIn("build_gate_promote_abliterated_m3.sh build", result.stderr)

    def test_quantized_module_is_rejected_before_mlx_import(self):
        class LowBitModule:
            scales = object()
            bits = 6
            group_size = 64

        with self.assertRaisesRegex(RuntimeError, "already-quantized M3 module"):
            self.helper.dequantize_layer_weight(LowBitModule())
        with self.assertRaisesRegex(RuntimeError, "already-quantized M3 module"):
            self.helper.requantize_layer(LowBitModule(), object(), 64, 6)

    def test_helper_contains_no_low_bit_round_trip_or_force_escape_hatch(self):
        source = self.script.read_text()
        force_flag = "--force" + "-quantized"
        self.assertNotIn("mx.dequantize(", source)
        self.assertNotIn("mx.quantize(", source)
        self.assertNotIn(force_flag, source)
        for path in ROOT.rglob("*"):
            if path.is_file() and ".git" not in path.parts \
                    and path.suffix in {".py", ".sh", ".md"}:
                self.assertNotIn(force_flag, path.read_text(), path)

    def test_m3_docs_name_only_the_bf16_then_single_quantization_path(self):
        docs = (ROOT / "README.md").read_text() + (ROOT / "M3_ABLITERATE_RUNBOOK.md").read_text()
        self.assertIn("fused_abliterate_quantize.py", docs)
        self.assertIn("full-VL bf16", docs)
        self.assertIn("quantize the complete", docs)
        self.assertNotIn("bake_abliteration_m3.py \\", docs)


if __name__ == "__main__":
    unittest.main()
