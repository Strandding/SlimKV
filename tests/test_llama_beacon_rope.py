"""CPU regressions for Beacon K RoPE during training, prefill and decode."""
import math
import unittest

import torch

from slimkv.llama3.configuration_llama import LlamaConfig
from slimkv.llama3.modeling_llama import LlamaAttention, repeat_kv
from slimkv.memory import Memory


def make_attention(**kwargs):
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        beacon_param=["q", "k", "v"],
        beacon_window=8,
        beacon_stride=8,
        beacon_ratio=[4],
        beacon_ratio_mix="sequence",
        beacon_sink_size=1,
        beacon_skip_rope_for_beacon_k=True,
        torch_dtype=torch.float32,
    )
    for name, value in kwargs.items():
        setattr(config, name, value)
    config._attn_implementation = "eager"
    attention = LlamaAttention(config, layer_idx=0)
    with torch.no_grad():
        # Nonzero Beacon projections make incorrect historical rotations observable.
        for parameter in attention.parameters():
            parameter.normal_(mean=0.0, std=0.1)
    return attention


class BeaconKeyRoPETest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def assert_rotary_rule(self, roles, q_len, indices, beacon_size, **config):
        attention = make_attention(**config)
        q = torch.randn(1, 4, q_len, 8, requires_grad=True)
        k = torch.randn(1, 2, len(roles), 8, requires_grad=True)
        original_k = k.detach().clone()
        positions = torch.arange(len(roles)).unsqueeze(0)
        expected_q, rotated_k = attention.rotary_emb(q, k, positions)
        mask = torch.tensor(roles, dtype=torch.bool)[None, None, :, None]
        expected_k = torch.where(mask, k, rotated_k)
        actual_q, actual_k = attention._apply_rotary_with_beacon(
            q, k, positions, beacon_size, indices
        )
        torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(k.detach(), original_k, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(actual_q.square().sum() + actual_k.sum(), (q, k), retain_graph=True)
        expected_grad = torch.autograd.grad(expected_q.square().sum() + expected_k.sum(), (q, k))
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_native_raw_path(self):
        self.assert_rotary_rule([False] * 4, 2, None, 0)

    def test_raw_only_beacon_path(self):
        self.assert_rotary_rule([False] * 3, 3, torch.zeros(3, dtype=torch.long), 0)

    def test_current_and_historical_beacons(self):
        # One sink, one historical Beacon, then raw / new Beacon / raw.
        self.assert_rotary_rule([False, True, False, True, False], 3, torch.tensor([0, 1, 0]), 1)

    def test_decode_without_new_beacons(self):
        # One sink, two historical Beacons, one raw tail, one new raw token.
        self.assert_rotary_rule([False, True, True, False, False], 1, torch.zeros(2, dtype=torch.long), 0)

    def test_rope_enabled_baseline(self):
        self.assert_rotary_rule([False] * 5, 1, torch.zeros(2, dtype=torch.long), 0,
                                beacon_skip_rope_for_beacon_k=False)

    def test_chunk_fixed_history_without_new_beacons(self):
        # Historical fixed-RoPE keys must not be rotated again at decode positions.
        self.assert_rotary_rule([False, True, True, False], 1, torch.zeros(1, dtype=torch.long), 0,
                                beacon_skip_rope_for_beacon_k=False,
                                beacon_chunk_fixed_rope_for_beacon_k=True)

    def run_memory_step(self, attention, memory):
        sink_count, history_count, _ = memory.get_memory_size()
        ids, mask, positions, past, _ = memory.step()
        past_k, _, beacon_size, indices = past[0]
        hidden = torch.randn(1, ids.shape[1], 32, requires_grad=memory.training)
        q, k, _ = attention.qkv_proj_with_beacon(hidden, beacon_size, indices)
        q = q.reshape(1, -1, 4, 8).transpose(1, 2)
        k = k.reshape(1, -1, 2, 8).transpose(1, 2)
        all_k = k if past_k is None else torch.cat([past_k, k], dim=2)
        q_rotated, k_rotated = attention.rotary_emb(q, all_k, positions)
        # Explicit cache roles, independent of the helper's early-return condition.
        beacon_roles = torch.cat([
            torch.zeros(sink_count, dtype=torch.bool),
            torch.ones(history_count, dtype=torch.bool),
            (indices == 1) | (indices == 2),
        ])
        self.assertEqual(beacon_roles.numel(), all_k.shape[2])
        expected_k = torch.where(beacon_roles[None, None, :, None], all_k, k_rotated)
        expected_weights = torch.softmax(
            q_rotated @ repeat_kv(expected_k, 2).transpose(2, 3) / math.sqrt(8) + mask,
            dim=-1,
        )
        output, weights, cache = attention(
            hidden, attention_mask=mask, position_ids=positions,
            past_key_value=past[0], output_attentions=True,
        )
        torch.testing.assert_close(weights, expected_weights, rtol=1e-6, atol=1e-7)
        # The persistent incremental K cache remains pre-RoPE.
        torch.testing.assert_close(cache[0], k, rtol=0, atol=0)
        memory.update_memory([cache])
        return beacon_size, history_count, output

    def test_training_final_window(self):
        attention = make_attention().train()
        memory = Memory(attention.config).train()
        ids = torch.arange(16).unsqueeze(0)
        memory.prepare(ids, torch.ones_like(ids), ids.clone())
        first_size, _, _ = self.run_memory_step(attention, memory)
        last_size, history, output = self.run_memory_step(attention, memory)
        self.assertGreater(first_size, 0)
        self.assertEqual(last_size, 0)
        self.assertGreater(history, 0)
        self.assertTrue(memory.finish)
        output.square().sum().backward()
        self.assertIsNotNone(attention.beacon_k_proj.weight.grad)
        self.assertTrue(torch.isfinite(attention.beacon_k_proj.weight.grad).all())

    def test_prefill_tail_and_decode(self):
        attention = make_attention().eval()
        memory = Memory(attention.config).eval()
        ids = torch.arange(10).unsqueeze(0)
        memory.prepare(ids, torch.ones_like(ids), None)
        with torch.no_grad():
            self.run_memory_step(attention, memory)
            tail_size, history, _ = self.run_memory_step(attention, memory)
            self.assertEqual(tail_size, 0)
            self.assertGreater(history, 0)
            self.assertTrue(memory.finish)
            memory.prepare(torch.tensor([[11]]), torch.ones(1, 1, dtype=torch.long), None)
            decode_size, history, _ = self.run_memory_step(attention, memory)
            self.assertEqual(decode_size, 0)
            self.assertGreater(history, 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
