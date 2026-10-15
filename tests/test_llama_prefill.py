"""Verify cache indexing and request cleanup independently of CUDA fusion."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from slimkv.llama3.prefill_ops import PrefillMetadata, fused_prefill_rope
from test_llama_reconstruction_cache import make_model


def cpu_metadata(model, ids, mask, positions, past, labels):
    roles = past[0][3]
    current = roles[-ids.shape[1]:]
    return PrefillMetadata((current == 0).nonzero().flatten(),
                           (current != 0).nonzero().flatten(), roles, positions,
                           bool((current == 2).any()))


class PrefillTest(unittest.TestCase):
    def test_fallback_invalidates_static_decode_without_prefill_metadata(self):
        model = make_model(beacon_decode_backend='latent', beacon_prefill_fast=False)
        model._static_decode = SimpleNamespace(epoch=object())
        with torch.inference_mode():
            model(input_ids=torch.arange(8)[None])
        self.assertIsNone(model._static_decode.epoch)

    def test_shared_indices_keep_partial_and_complete_window_caches_exact(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        original = make_model(beacon_decode_backend='latent').to(torch.bfloat16)
        candidate = copy.deepcopy(original)
        cursor = 0
        with torch.inference_mode():
            for count in [3, 5, 9, 2]:
                ids = torch.arange(cursor, cursor + count)[None]
                expected = original(input_ids=ids).logits
                with patch('slimkv.llama3.prefill_ops.prepare_metadata', cpu_metadata):
                    actual = candidate(input_ids=ids).logits
                self.assertTrue(torch.equal(expected, actual))
                for group in ('sink_activations', 'raw_activations', 'beacon_activations', 'window_beacon_activations'):
                    for left, right in zip(getattr(original.memory, group), getattr(candidate.memory, group)):
                        for a, b in zip(left, right):
                            self.assertEqual(a is None, b is None)
                            if a is not None:
                                self.assertTrue(torch.equal(a, b), group)
                self.assertFalse(candidate._prefill_metadata_active)
                self.assertTrue(all(layer.self_attn._prefill_metadata is None for layer in candidate.model.layers))
                cursor += count

    def test_interrupted_forward_drops_temporary_metadata(self):
        model = make_model(beacon_decode_backend='latent')
        with torch.inference_mode(), patch('slimkv.llama3.prefill_ops.prepare_metadata', cpu_metadata):
            with patch.object(model.model.layers[0], 'forward', side_effect=RuntimeError('test abort')):
                with self.assertRaisesRegex(RuntimeError, 'test abort'):
                    model(input_ids=torch.arange(8)[None])
        self.assertFalse(model._prefill_metadata_active)
        self.assertTrue(all(layer.self_attn._prefill_metadata is None for layer in model.model.layers))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for rotary fusion')
class PrefillRotaryCudaTest(unittest.TestCase):
    def test_bf16_rounding_strides_history_and_sink(self):
        try:
            import triton
        except ImportError:
            self.skipTest('Triton is required for rotary fusion')
        torch.manual_seed(42)
        rope = make_model().cuda().to(torch.bfloat16).model.layers[0].self_attn.rotary_emb
        for batch, qlen, length, contiguous, sink in [
                (1, 3, 3, False, 0), (2, 37, 129, False, 1),
                (4, 32, 65, True, 3)]:
            q = torch.randn(batch, qlen, 4, 8, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
            k = torch.randn(batch, length, 2, 8, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
            if contiguous:
                q, k = q.contiguous(), k.contiguous()
            roles = (torch.arange(qlen, device='cuda') % 5 == 4).long()
            roles[::17] = 2
            positions = torch.arange(length, device='cuda')[None].expand(batch, -1)
            metadata = PrefillMetadata(None, None, roles, positions, True)
            expected_q, expected_k = rope(q, k, positions)
            prefix = length - qlen
            prefix_roles = torch.arange(prefix, device='cuda') >= sink
            mask = torch.cat([prefix_roles, roles != 0])
            expected_k[:, :, mask] = k[:, :, mask]
            actual_q, actual_k = fused_prefill_rope(q, k, rope, metadata, sink)
            self.assertTrue(torch.equal(expected_q.contiguous().view(torch.int16),
                                        actual_q.contiguous().view(torch.int16)))
            self.assertTrue(torch.equal(expected_k.contiguous().view(torch.int16),
                                        actual_k.contiguous().view(torch.int16)))


if __name__ == '__main__':
    unittest.main()
