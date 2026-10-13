"""The compact full-cache timing control must preserve inference values."""
import copy
import unittest

import torch

from tools.benchmark_reconstruction import install_compact_sink_control
from test_llama_reconstruction_cache import make_model


class CompactSinkControlTest(unittest.TestCase):
    def test_compact_full_preserves_outputs_and_owns_sink_storage(self):
        torch.manual_seed(123)
        stock = make_model(beacon_cache_mode='full')
        compact = copy.deepcopy(stock)
        install_compact_sink_control(compact.memory, {'compact_sink': True})
        with torch.no_grad():
            for ids in [torch.arange(19)[None], torch.tensor([[20]]), torch.tensor([[21]])]:
                a = stock(input_ids=ids)
                b = compact(input_ids=ids)
                torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
                self.assertEqual(stock.memory.cache_bytes(), compact.memory.cache_bytes())
                for actual, expected in zip(compact.memory.sink_activations, stock.memory.sink_activations):
                    for x, y in zip(actual, expected):
                        torch.testing.assert_close(x, y, rtol=0, atol=0)
                        self.assertEqual(x.untyped_storage().nbytes(), x.numel() * x.element_size())
                for group_a, group_b in [(compact.memory.beacon_activations, stock.memory.beacon_activations),
                                        (compact.memory.raw_activations, stock.memory.raw_activations)]:
                    for pair_a, pair_b in zip(group_a, group_b):
                        for x, y in zip(pair_a, pair_b):
                            if x is not None:
                                torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_control_is_inactive_for_latent_cache(self):
        torch.manual_seed(123)
        original = make_model()
        controlled = copy.deepcopy(original)
        install_compact_sink_control(controlled.memory, {'compact_sink': True})
        with torch.no_grad():
            ids = torch.arange(19)[None]
            a = original(input_ids=ids)
            b = controlled(input_ids=ids)
            torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
            self.assertTrue(controlled.memory.latent_cache_active)
            self.assertEqual(original.memory.cache_bytes(), controlled.memory.cache_bytes())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
