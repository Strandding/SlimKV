"""Check optional compilation fallbacks and the performance preset's arguments."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from slimkv.llama3 import compile_decode
from test_llama_reconstruction_cache import make_model

ROOT = Path(__file__).resolve().parents[1]


class CompileDecodeTest(unittest.TestCase):
    def test_cpu_training_keeps_outputs_and_gradients(self):
        torch.manual_seed(123)
        model = make_model(beacon_decode_compile=True).train()
        layer = model.model.layers[0]
        attention = torch.randn(1, 2, 32, requires_grad=True)
        residual = torch.randn_like(attention, requires_grad=True)
        with patch.object(compile_decode, 'compiled_tail', side_effect=AssertionError('must stay eager')):
            actual = layer._decode_tail(attention, residual)
        model.config.beacon_decode_compile = False
        expected = layer._decode_tail(attention, residual)
        self.assertTrue(torch.equal(actual, expected))
        inputs = (attention, residual, *layer.mlp.parameters(), *layer.post_attention_layernorm.parameters())
        a = torch.autograd.grad(actual.sum(), inputs, retain_graph=True)
        b = torch.autograd.grad(expected.sum(), inputs)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))

    def test_unvalidated_torch_version_does_not_compile(self):
        compile_decode.compiled_tail.cache_clear()
        try:
            with patch.object(torch, '__version__', '2.5.1'), patch.object(torch, 'compile') as compiler:
                with self.assertWarnsRegex(UserWarning, 'Torch 2.11'):
                    self.assertIsNone(compile_decode.compiled_tail())
                compiler.assert_not_called()
        finally:
            compile_decode.compiled_tail.cache_clear()

    def test_longbench_fast_defaults_and_explicit_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp) / 'python'
            python.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            python.chmod(0o755)
            base = dict(PATH=os.environ['PATH'], PYTHON=str(python), MODEL_NAME_OR_PATH=tmp)
            for overrides, expected in [
                ({}, ('flash_attention_2', 'latent', 'True', 'True')),
                ({'ATTN_IMPL': 'eager'}, ('eager', 'reconstruct', 'False', 'False')),
                ({'DECODE_COMPILE': 'False'}, ('flash_attention_2', 'latent', 'True', 'False')),
            ]:
                run = subprocess.run(['bash', 'scripts/evaluation/eval_longbench_slimkv.sh'],
                                     cwd=ROOT, env={**base, **overrides}, text=True,
                                     capture_output=True, check=True)
                args = json.loads(run.stdout)
                value = lambda flag: args[args.index(flag) + 1]
                actual = tuple(value(flag) for flag in ('--attn_impl', '--beacon_decode_backend',
                                                        '--beacon_decode_graph', '--beacon_decode_compile'))
                self.assertEqual(actual, expected)
                self.assertEqual(value('--beacon_decode_graph_full'), 'True')
                self.assertEqual(value('--beacon_prefill_fast'), 'True')
                self.assertEqual(value('--batch_size'), '32')


if __name__ == '__main__':
    unittest.main()
