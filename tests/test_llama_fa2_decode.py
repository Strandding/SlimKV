"""Validate global latent algebra and causal decode transitions in FP32."""
import copy
import math
import types
import unittest
from unittest.mock import patch

import torch

from test_llama_reconstruction_cache import make_model
from slimkv.llama3.fa2_decode import merge_attention, flash_forward


def reference_forward(q, k, v, scale, causal=False):
    k = k.repeat_interleave(q.shape[2] // k.shape[2], dim=2)
    v = v.repeat_interleave(q.shape[2] // v.shape[2], dim=2)
    logits = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) * scale
    if causal:
        mask = torch.arange(k.shape[1])[None] > torch.arange(q.shape[1])[:, None] + k.shape[1] - q.shape[1]
        logits = logits.masked_fill(mask, -math.inf)
    lse = logits.logsumexp(-1)
    probs = torch.softmax(logits, -1).nan_to_num()
    return (probs @ v.transpose(1, 2)).transpose(1, 2), lse


class LatentDecodeTest(unittest.TestCase):
    def test_actual_models_across_new_beacons_and_window_rollover(self):
        torch.set_num_threads(1)
        for bias, shared in [(False, False), (True, False), (False, True)]:
            torch.manual_seed(42)
            settings = {'attention_bias': bias, 'beacon_shared_kv_down_proj': shared}
            if shared:
                settings['beacon_v_latent_head_dim_list'] = [2, 3]
            baseline = make_model(**settings)
            latent = copy.deepcopy(baseline)
            latent.config.beacon_decode_backend = 'fa2'
            with torch.inference_mode(), patch('slimkv.llama3.fa2_decode.flash_forward', side_effect=reference_forward) as kernel:
                # Includes a one-token prefill tail, which must stay on the full path.
                ids = torch.arange(17)[None]
                a, b = baseline(input_ids=ids), latent(input_ids=ids)
                torch.testing.assert_close(a.logits, b.logits, atol=2e-6, rtol=2e-5)
                self.assertEqual(kernel.call_count, 0)
                for token in range(18, 38):
                    ids = torch.tensor([[token]])
                    a, b = baseline(input_ids=ids), latent(input_ids=ids)
                    torch.testing.assert_close(a.logits, b.logits, atol=2e-6, rtol=2e-5)
                    self.assertFalse(latent.config._slimkv_decode_step)
                self.assertGreater(kernel.call_count, 20)
                for k, v in latent.memory.beacon_activations:
                    self.assertEqual(k.shape[1], 1)
                    self.assertEqual(v.shape[1], 1)
                self.assertEqual(sum(baseline.memory.cache_bytes().values()), sum(latent.memory.cache_bytes().values()))

    def test_softmax_merge_matches_single_joint_distribution(self):
        torch.manual_seed(1)
        q = torch.randn(2, 3, 4, 8)
        k, v = torch.randn(2, 11, 2, 8), torch.randn(2, 11, 2, 8)
        x, xl = reference_forward(q, k[:, :5], v[:, :5], 1 / math.sqrt(8))
        y, yl = reference_forward(q, k[:, 5:], v[:, 5:], 1 / math.sqrt(8))
        expected, _ = reference_forward(q, k, v, 1 / math.sqrt(8))
        merged = merge_attention(x.transpose(1, 2), xl, y.transpose(1, 2), yl).transpose(1, 2)
        torch.testing.assert_close(merged, expected)

    def test_fully_masked_fa2_row_has_zero_merge_mass(self):
        def native(q, k, v, scale, causal):
            lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1])
            lse[..., 0] = math.inf
            return torch.zeros_like(q), lse

        extension = types.SimpleNamespace(forward=native)
        with patch.dict('sys.modules', {'slimkv_fa2_decode_320': extension}):
            q = torch.ones(1, 2, 4, 304)
            k, v = torch.ones(1, 1, 1, 304), torch.ones(1, 1, 1, 312)
            output, lse = flash_forward(q, k, v, 1 / math.sqrt(128), True)
            self.assertTrue(lse[..., 0].isneginf().all())
            raw = torch.randn(1, 4, 2, 8)
            merged = merge_attention(raw, torch.zeros(1, 4, 2), output[..., :8].transpose(1, 2), lse)
            torch.testing.assert_close(merged[:, :, 0], raw[:, :, 0])

    def test_short_prefill_first_beacon_with_empty_latent_history(self):
        baseline = make_model()
        latent = copy.deepcopy(baseline)
        latent.config.beacon_decode_backend = 'fa2'
        with torch.inference_mode(), patch('slimkv.llama3.fa2_decode.flash_forward', side_effect=reference_forward):
            for ids in [torch.arange(3)[None]] + [torch.tensor([[i]]) for i in range(4, 18)]:
                a, b = baseline(input_ids=ids), latent(input_ids=ids)
                torch.testing.assert_close(a.logits, b.logits, atol=2e-6, rtol=2e-5)

    def test_incompatible_configuration_fails_explicitly(self):
        for kwargs in [{'beacon_cache_mode': 'full'}, {'beacon_skip_rope_for_beacon_k': False},
                       {'beacon_attend_prev': False}]:
            model = make_model(beacon_decode_backend='fa2', **kwargs)
            with self.assertRaises(ValueError):
                model(input_ids=torch.arange(17)[None])


if __name__ == '__main__':
    unittest.main()
