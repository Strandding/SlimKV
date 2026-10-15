"""Decode-only compression control, prompt preservation and unbounded raw growth."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_llama_reconstruction_cache import make_model
from test_llama_fa2_decode import reference_forward
from slimkv.llama3.memory import LlamaMemory
from slimkv.memory import Memory
from tools.benchmark.benchmark_context_decode import cloned_memory, validate_runtime_cache
from tools.benchmark.context_decode_runtime import OptimizedDecodeRuntime

ROOT = Path(__file__).resolve().parents[1]


def memories(memory):
    return {name: [tuple(t.clone() if t is not None else None for t in pair)
                   for pair in getattr(memory, name)]
            for name in ['beacon_activations', 'window_beacon_activations', 'sink_activations']
            if hasattr(memory, name)}


class DecodeCompressionTest(unittest.TestCase):
    def assert_memory_equal(self, expected, memory):
        for name, pairs in expected.items():
            for left, right in zip(pairs, getattr(memory, name)):
                for x, y in zip(left, right):
                    self.assertTrue((x is None and y is None) or
                                    (x is not None and y is not None and torch.equal(x, y)), name)

    def test_default_is_bitwise_compatible_and_prefill_is_independent(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        enabled = make_model(beacon_decode_compress=True)
        legacy, disabled = copy.deepcopy(enabled), copy.deepcopy(enabled)
        del legacy.config.beacon_decode_compress
        disabled.config.beacon_decode_compress = False
        with torch.inference_mode():
            prefix = torch.arange(19)[None]
            expected = enabled(input_ids=prefix).logits
            self.assertTrue(torch.equal(expected, legacy(input_ids=prefix).logits))
            self.assertTrue(torch.equal(expected, disabled(input_ids=prefix).logits))
            self.assert_memory_equal(memories(enabled.memory), disabled.memory)
            for token in range(20, 38):
                ids = torch.tensor([[token]])
                self.assertTrue(torch.equal(enabled(input_ids=ids).logits, legacy(input_ids=ids).logits))

    def test_decode_keeps_prompt_memories_and_grows_raw_across_windows(self):
        torch.set_num_threads(1)
        for prefix in [3, 16, 19]:
            for full_cache in [False, True]:
                model = make_model(beacon_decode_compress=False)
                if full_cache:
                    model.config.beacon_cache_mode = 'full'
                with torch.inference_mode():
                    model(input_ids=torch.arange(prefix)[None])
                    saved = memories(model.memory)
                    raw = 0 if model.memory.raw_activations[0][0] is None else model.memory.raw_activations[0][0].shape[2]
                    for step in range(20):
                        model(input_ids=torch.tensor([[(prefix+step) % 64]]))
                        self.assertEqual(model.memory._all_beacon_sizes[-1], 0)
                        self.assertEqual(model.memory.raw_activations[0][0].shape[2], raw+step+1)
                        self.assert_memory_equal(saved, model.memory)
                    model.config.beacon_decode_compress = True
                    with self.assertRaisesRegex(RuntimeError, 'Reset Memory'):
                        model(input_ids=torch.tensor([[42]]))
                    model.memory.reset()
                    model(input_ids=torch.arange(16)[None])
                    self.assertFalse(model.memory._decode_uncompressed)

    def test_fast_metadata_matches_ordinary_with_long_raw_continuations(self):
        torch.manual_seed(9)
        slow = make_model(beacon_decode_backend='fa2', beacon_decode_fast=False,
                          beacon_decode_compress=False)
        fast = copy.deepcopy(slow)
        fast.config.beacon_decode_fast = True
        with torch.inference_mode(), patch('slimkv.llama3.fa2_decode.flash_forward', side_effect=reference_forward):
            for ids in [torch.arange(19)[None]] + [torch.tensor([[i]]) for i in range(20, 50)]:
                a, b = slow(input_ids=ids), fast(input_ids=ids)
                torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
                self.assertEqual(slow.memory.get_memory_size(), fast.memory.get_memory_size())
            self.assertGreater(fast.memory._decode_role_buffer.numel(), 2*fast.config.beacon_window+2)

    def test_optimized_benchmark_matches_no_compression_reference(self):
        torch.manual_seed(7)
        for backend in ['reconstruct', 'latent']:
            model = make_model(beacon_decode_backend=backend, beacon_decode_compress=False)
            reference = copy.deepcopy(model)
            with torch.inference_mode():
                prefix = torch.arange(16)[None]
                model(input_ids=prefix)
                reference(input_ids=prefix)
                runtime = OptimizedDecodeRuntime(model, backend, 19, memory=cloned_memory(model.memory), graph=False)
                for token in range(20, 39):
                    ids = torch.tensor([[token]])
                    expected = reference(input_ids=ids).logits[:, -1:]
                    torch.testing.assert_close(runtime.run(ids), expected, rtol=2e-5, atol=2e-6)
                # FP32 eager and fused algebra can differ in their last bits;
                # the real BF16 CUDA checks require bitwise equality separately.
                self.assertLessEqual(validate_runtime_cache(runtime, reference.memory)['max_error'], 2e-6)

    def test_qwen_shared_scheduler_keeps_decode_tokens_raw(self):
        from slimkv.qwen2.configuration_qwen2 import Qwen2Config
        from slimkv.qwen2.modeling_qwen2 import Qwen2ForCausalLM
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=48,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=128, torch_dtype=torch.float32,
                            beacon_param=['q', 'k', 'v'], beacon_window=8, beacon_stride=8,
                            beacon_ratio=[4], beacon_ratio_mix='sequence', beacon_sink_size=1,
                            beacon_decode_compress=False)
        config._attn_implementation = 'eager'
        model = Qwen2ForCausalLM(config)
        model.memory = Memory(config)
        model.eval()
        with torch.inference_mode():
            model(input_ids=torch.arange(19)[None])
            saved = memories(model.memory)
            raw = model.memory.raw_activations[0][0].shape[2]
            for token in range(20, 40):
                model(input_ids=torch.tensor([[token]]))
                self.assertEqual(model.memory._all_beacon_sizes[-1], 0)
                self.assert_memory_equal(saved, model.memory)
            self.assertEqual(model.memory.raw_activations[0][0].shape[2], raw+20)

    def test_training_ignores_inference_compression_setting(self):
        torch.manual_seed(5)
        enabled = make_model(beacon_decode_compress=True).train()
        disabled = copy.deepcopy(enabled)
        disabled.config.beacon_decode_compress = False
        ids = torch.arange(19)[None]
        a, b = enabled(input_ids=ids, labels=ids), disabled(input_ids=ids, labels=ids)
        self.assertTrue(torch.equal(a.logits, b.logits))
        self.assertTrue(torch.equal(a.loss, b.loss))
        self.assertEqual(enabled.memory._all_beacon_sizes, disabled.memory._all_beacon_sizes)
        self.assertTrue(any(size > 0 for size in disabled.memory._all_beacon_sizes))

    def test_launchers_forward_flag_and_separate_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp)/'python'
            python.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            python.chmod(0o755)
            env = dict(PATH=os.environ['PATH'], PYTHON=str(python), MODEL_NAME_OR_PATH=tmp,
                       DECODE_COMPRESS='False')
            for script, flag, suffix in [
                ('scripts/evaluation/eval_longbench_slimkv.sh', '--beacon_decode_compress', '-rawdecode'),
                ('scripts/benchmark/benchmark_context_decode.sh', '--no-beacon_decode_compress', '_raw_decode')]:
                run = subprocess.run(['bash', script], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
                args = json.loads(run.stdout)
                self.assertIn(flag, args)
                self.assertTrue(any(arg.endswith(suffix) for arg in args))


if __name__ == '__main__':
    unittest.main()
