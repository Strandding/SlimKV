"""Verify the two query visibility sets and shared-KV layout independently of FA2."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from slimkv.llama3.paired_decode import original_splits, paired_attention
from slimkv.llama3.latent_decode import eager_forward
from slimkv.llama3.static_decode import StaticDecode


class PairedDecodeTest(unittest.TestCase):
    def test_visibility_and_shared_storage_for_mqa_and_gqa(self):
        torch.manual_seed(17)
        for kv_heads in [1, 8]:
            q = torch.randn(1, 2, 32, 8)
            k, v = torch.randn(1, 9, kv_heads, 8), torch.randn(1, 9, kv_heads, 8)
            lengths = torch.tensor([8, 9], dtype=torch.int32)
            seen = []
            def forward(a, b, c, scale, causal, splits, pack, lens):
                self.assertFalse(causal)
                self.assertEqual(b.stride(0), 0)
                self.assertEqual(b.data_ptr(), k.data_ptr())
                seen.append(lens.tolist())
                pairs = [eager_forward(a[i:i+1], b[i:i+1, :int(lens[i])],
                                       c[i:i+1, :int(lens[i])], scale, False) for i in range(2)]
                return torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])
            actual, lse = paired_attention(forward, q, k, v, 0.5, lengths, 78)
            expected, expected_lse = eager_forward(q, k, v, 0.5, True)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(lse, expected_lse)
            self.assertEqual(seen, [[8, 9]])

    def test_split_counts_match_unpacked_q2(self):
        for length in [8256, 16448, 32832]:
            self.assertEqual(original_splits(320, length, 32, 78), 3)
            self.assertEqual(original_splits(256, length, 32, 78), 9)

    def test_static_gate_accepts_a_raw_token_with_its_beacon(self):
        rope = type('Llama3RotaryEmbedding', (), {'max_seq_len_cached': 4096})()
        attn = SimpleNamespace(beacon_k_up_proj=torch.nn.Linear(208, 16),
                               beacon_v_up_proj=torch.nn.Linear(208, 16), rotary_emb=rope)
        model = SimpleNamespace(config=SimpleNamespace(
            beacon_decode_backend='fa2', beacon_decode_graph=True,
            beacon_decode_graph_full=True, beacon_sink_size=1), dtype=torch.bfloat16,
            memory=SimpleNamespace(_fast_decode_active=True),
            model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]))
        past = [(torch.empty(1, 1, 20, 16), None, 1, None)]
        with torch.inference_mode(), patch('slimkv.llama3.static_decode.cache_extension', return_value=object()):
            for length in [1, 2]:
                self.assertTrue(StaticDecode.eligible(model, SimpleNamespace(is_cuda=True, shape=(1, length)), past, {}))
            self.assertFalse(StaticDecode.eligible(model, SimpleNamespace(is_cuda=True, shape=(1, 3)), past, {}))


if __name__ == '__main__':
    unittest.main()
