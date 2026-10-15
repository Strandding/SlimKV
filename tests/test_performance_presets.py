"""Verify hardware selection, task policies and explicit launcher overrides."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from tools.performance_presets import resolve_preset, configure_longbench

ROOT = Path(__file__).resolve().parents[1]


class PerformancePresetTest(unittest.TestCase):
    def test_hardware_selection_and_manual_fallback(self):
        self.assertEqual(resolve_preset('auto', (9, 0)), 'sm90')
        self.assertEqual(resolve_preset('auto', (12, 0)), 'sm120')
        self.assertEqual(resolve_preset('auto', (8, 0)), 'manual')
        self.assertEqual(resolve_preset('manual'), 'manual')
        with self.assertRaises(ValueError):
            resolve_preset('unknown')

    def test_short_answer_policy_and_explicit_overrides(self):
        config = SimpleNamespace(beacon_decode_graph=True, beacon_decode_compile=False, ranks=[197, 313])
        configure_longbench(config, 'sm120', 'qasper')
        self.assertFalse(config.beacon_decode_graph)
        self.assertTrue(config.beacon_decode_compile)
        configure_longbench(config, 'sm120', 'qmsum')
        self.assertTrue(config.beacon_decode_graph)
        config.beacon_decode_graph = False
        config.beacon_decode_compile = False
        configure_longbench(config, 'sm120', 'qmsum', {'beacon_decode_graph', 'beacon_decode_compile'})
        self.assertFalse(config.beacon_decode_graph)
        self.assertFalse(config.beacon_decode_compile)
        configure_longbench(config, 'sm90', 'qasper')
        self.assertTrue(config.beacon_decode_graph)
        self.assertEqual(config.ranks, [197, 313])
        configure_longbench(config, 'manual', 'qasper')
        self.assertTrue(config.beacon_decode_graph)

    def test_launcher_defers_defaults_but_forwards_explicit_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / 'capture'
            script.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            script.chmod(0o755)
            env = {key: value for key, value in os.environ.items()
                   if key not in ['DECODE_GRAPH', 'DECODE_COMPILE', 'PERFORMANCE_PRESET']}
            env.update(PYTHON=str(script), MODEL_NAME_OR_PATH='checkpoint')
            command = ['bash', str(ROOT / 'scripts/evaluation/eval_longbench_slimkv.sh')]
            args = json.loads(subprocess.check_output(command, env=env, text=True))
            self.assertEqual(args[args.index('--performance_preset') + 1], 'auto')
            self.assertNotIn('--beacon_decode_graph', args)
            self.assertNotIn('--beacon_decode_compile', args)
            env.update(DECODE_GRAPH='False', DECODE_COMPILE='False')
            args = json.loads(subprocess.check_output(command, env=env, text=True))
            self.assertEqual(args[args.index('--beacon_decode_graph') + 1], 'False')
            self.assertEqual(args[args.index('--beacon_decode_compile') + 1], 'False')

    def test_csv_records_effective_runtime_and_actual_batches(self):
        from main.evaluation.eval_longbench import save_task_timings
        with tempfile.TemporaryDirectory() as tmp:
            save_task_timings(tmp, [{'task': 'qasper', 'score': 42,
                'batch_size_counts': {'1': 7, '2': 3}, 'performance_preset': 'sm120',
                'decode_graph': False, 'decode_compile': True}])
            with (Path(tmp) / 'task_timings.csv').open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(json.loads(row['batch_size_counts']), {'1': 7, '2': 3})
            self.assertEqual(row['performance_preset'], 'sm120')
            self.assertEqual(row['decode_graph'], 'False')


if __name__ == '__main__':
    unittest.main()
