"""Check page boundaries, request isolation and exact BF16 K/V storage."""
import unittest
import copy
from types import SimpleNamespace

import torch

from slimkv.vllm_plugin.paged_cache import BeaconPageLayout, BeaconPages, RawPages
from slimkv.vllm_plugin.paged_model import restore_memory, flush_memory, release_workspace_views
from test_llama_reconstruction_cache import make_model


class PagedCacheTest(unittest.TestCase):
    def test_stateless_restore_matches_model_through_windows_and_chunked_prefill(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        original = make_model(beacon_decode_backend='latent', torch_dtype=torch.bfloat16).to(torch.bfloat16)
        paged = copy.deepcopy(original)
        raw_layers, beacon_layers = [], []
        for layer in paged.model.layers:
            attn = layer.self_attn
            layout = BeaconPageLayout(16, 4, attn.beacon_k_up_proj.in_features,
                attn.beacon_v_up_proj.in_features, attn.num_key_value_heads * attn.head_dim)
            raw_layers.append(SimpleNamespace(kv_cache=torch.zeros(16, 4, 2, 2, 8, dtype=torch.bfloat16)))
            beacon_layers.append(SimpleNamespace(spec=SimpleNamespace(block_size=16),
                kv_cache=torch.zeros(16, layout.storage_rows, layout.width, dtype=torch.bfloat16)))
        raw_table = torch.tensor([9, 3, 12, 5, 1, 7, 8, 2, 6, 4])
        beacon_table = torch.tensor([7, 2, 8])
        cursor = 0
        with torch.inference_mode():
            for count in [3, 7, 7] + [1] * 17:
                ids = torch.arange(cursor, cursor + count)[None]
                expected = original(input_ids=ids).logits[:, -1]
                restore_memory(paged, cursor, raw_layers, beacon_layers, raw_table, beacon_table)
                if cursor:
                    # Historical token IDs must never be embedded again.
                    paged.memory.all_input_ids.fill_(999)
                actual = paged(input_ids=ids).logits[:, -1]
                self.assertTrue(torch.equal(actual, expected), f'cursor={cursor}, count={count}')
                flush_memory(paged, cursor, cursor + count, raw_layers, beacon_layers, raw_table, beacon_table)
                release_workspace_views(paged.memory)
                self.assertEqual(sum(paged.memory.cache_bytes().values()), 0)
                cursor += count

    def test_shuffled_pages_and_partial_appends_keep_distinct_unaligned_ranks(self):
        layout = BeaconPageLayout(4, 2, 3, 5, 4)
        backing = torch.full((12, layout.page_elements + 7), -13, dtype=torch.bfloat16)
        pages = BeaconPages(backing[:, :layout.page_elements], layout)
        tables = [torch.tensor([7, 2, 9]), torch.tensor([4, 1, 10])]
        stored = []
        for index, table in enumerate(tables):
            k = torch.arange(18).reshape(1, 1, 6, 3).to(torch.bfloat16) + index * 32
            v = torch.arange(30).reshape(1, 1, 6, 5).to(torch.bfloat16) + index * 32
            pages.append(table, 0, k[:, :, :3], v[:, :, :3])
            pages.append(table, 3, k[:, :, 3:], v[:, :, 3:])
            sink = torch.full((1, 1, 1, 4), index + 1, dtype=torch.bfloat16)
            pages.write_sink(table[0], sink, sink + 2)
            stored.append((k, v, sink))
        for table, (k, v, sink) in zip(tables, stored):
            actual_k, actual_v = pages.read(table, 6)
            self.assertTrue(torch.equal(actual_k, k))
            self.assertTrue(torch.equal(actual_v, v))
            sk, sv = pages.read_sink(table[0], 1, 4)
            self.assertTrue(torch.equal(sk, sink))
            self.assertTrue(torch.equal(sv, sink + 2))
        self.assertTrue((backing[:, layout.page_elements:] == -13).all())
        self.assertEqual(pages.read(tables[0], 0), (None, None))

    def test_raw_window_uses_absolute_positions_and_reused_physical_pages(self):
        raw = RawPages(torch.zeros(8, 4, 2, 2, 3, dtype=torch.bfloat16))
        table = torch.tensor([5, 2, 6, 1])
        key = torch.arange(42).reshape(1, 2, 7, 3).to(torch.bfloat16)
        value = -key
        raw.write(table, 5, key, value)
        actual = raw.read(table, 5, 7)
        self.assertTrue(torch.equal(actual[0], key))
        self.assertTrue(torch.equal(actual[1], value))
        other = torch.tensor([5, 2, 6, 1])
        raw.write(other, 5, key + 1, value - 1)
        actual = raw.read(other, 5, 7)
        self.assertTrue(torch.equal(actual[0], key + 1))
        self.assertTrue(torch.equal(actual[1], value - 1))


if __name__ == '__main__':
    unittest.main()
