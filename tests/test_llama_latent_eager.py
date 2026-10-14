"""Validate eager latent attention without rebuilding historical Beacon K/V."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_llama_reconstruction_cache import make_model
from test_llama_fa2_decode import reference_forward
from slimkv.llama3.latent_decode import eager_forward, uses_fa2_decode
from slimkv.llama3.memory import ReconstructionCache


class LatentEagerTest(unittest.TestCase):
    def test_attention_reference_precision_survives_autocast(self):
        torch.manual_seed(42)
        q, k, v = torch.randn(1, 2, 4, 13), torch.randn(1, 7, 2, 13), torch.randn(1, 7, 2, 9)
        expected, expected_lse = eager_forward(q, k, v, 0.125, True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual, lse = eager_forward(q, k, v, 0.125, True)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(lse, expected_lse, atol=0, rtol=0)

    def test_grouped_attention_handles_distinct_ranks_and_masked_rows(self):
        torch.manual_seed(42)
        for queries, keys, rank_k, rank_v in [(1, 7, 13, 9), (2, 7, 13, 17), (2, 1, 384, 392)]:
            q = torch.randn(2, queries, 4, rank_k)
            k = torch.randn(2, keys, 2, rank_k)
            v = torch.randn(2, keys, 2, rank_v)
            for causal in [False, True]:
                actual, lse = eager_forward(q, k, v, 0.125, causal)
                expected, expected_lse = reference_forward(q, k, v, 0.125, causal)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(lse, expected_lse, atol=2e-6, rtol=2e-5)
                self.assertFalse(actual.isnan().any())

    def test_models_match_reconstruction_without_materializing_decode_cache(self):
        torch.set_num_threads(1)
        for batch, prefix, bias, shared in [(1, 3, False, False), (3, 17, True, False), (2, 17, False, True)]:
            torch.manual_seed(42)
            settings = dict(attention_bias=bias, beacon_shared_kv_down_proj=shared)
            if shared:
                settings['beacon_v_latent_head_dim_list'] = [2, 3]
            baseline = make_model(**settings)
            eager = copy.deepcopy(baseline)
            eager.config.beacon_decode_backend = 'latent'
            fa2 = copy.deepcopy(baseline)
            fa2.config.beacon_decode_backend = 'fa2'
            fa2.config.beacon_decode_fast = False
            offsets = torch.arange(batch)[:, None] * 3
            with torch.inference_mode(), patch('slimkv.llama3.fa2_decode.flash_forward', side_effect=reference_forward):
                ids = (torch.arange(prefix)[None] + offsets) % 64
                a, b, c = [model(input_ids=ids) for model in [baseline, eager, fa2]]
                torch.testing.assert_close(a.logits, b.logits, atol=3e-6, rtol=2e-5)
                for token in range(prefix + 1, prefix + 22):
                    ids = (torch.tensor([[token]]) + offsets) % 64
                    a, c = baseline(input_ids=ids), fa2(input_ids=ids)
                    with patch.object(ReconstructionCache, 'materialize', side_effect=AssertionError('Decode rebuilt K/V')):
                        b = eager(input_ids=ids)
                    torch.testing.assert_close(a.logits, b.logits, atol=3e-6, rtol=2e-5)
                    torch.testing.assert_close(c.logits, b.logits, atol=3e-6, rtol=2e-5)
                    self.assertEqual(eager.memory.get_memory_size(), baseline.memory.get_memory_size())
                self.assertEqual(sum(eager.memory.cache_bytes().values()), sum(baseline.memory.cache_bytes().values()))
                for k, v in eager.memory.beacon_activations:
                    self.assertEqual(k.shape[1], 1)
                    self.assertEqual(v.shape[1], 1)

    def test_eager_has_no_flash_dependency_and_diagnostics_use_reconstruction(self):
        model = make_model(beacon_decode_backend='latent')
        with torch.inference_mode(), patch('slimkv.llama3.fa2_decode.flash_forward', side_effect=AssertionError('Used FlashAttention')):
            model(input_ids=torch.arange(17)[None])
            model(input_ids=torch.tensor([[18]]))
            with patch('slimkv.llama3.fa2_decode.eager_forward', side_effect=AssertionError('Ignored diagnostics')):
                output = model(input_ids=torch.tensor([[19]]), output_attentions=True)
            self.assertIsNotNone(output.attentions)
            with self.assertRaisesRegex(ValueError, 'unpadded'):
                model(input_ids=torch.tensor([[20]]), attention_mask=torch.zeros(1, 1))

    def test_latent_strategy_selects_attention_implementation(self):
        for backend, implementation, expected in [
            ('latent', 'eager', False), ('latent', 'flash_attention_2', True),
            ('fa2', 'eager', True), ('reconstruct', 'eager', False),
        ]:
            config = SimpleNamespace(beacon_decode_backend=backend, _attn_implementation=implementation)
            self.assertEqual(uses_fa2_decode(config), expected)


if __name__ == '__main__':
    unittest.main()
