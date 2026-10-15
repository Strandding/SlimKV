"""Check shared optimized decode against independent model/cache implementations."""
import copy
import unittest

import torch

from test_llama_reconstruction_cache import make_model
from tools.benchmark.benchmark_context_decode import AppendCache, cloned_memory, full_backbone
from tools.benchmark.context_decode_runtime import OptimizedDecodeRuntime


class OptimizedRuntimeTest(unittest.TestCase):
    def test_slim_paths_match_through_beacon_insertion_and_cache_reset(self):
        torch.set_num_threads(1)
        for backend in ['reconstruct', 'latent']:
            for bias in [False, True]:
                torch.manual_seed(42)
                model = make_model(beacon_decode_backend=backend, attention_bias=bias)
                reference = copy.deepcopy(model)
                with torch.inference_mode():
                    prefix = torch.arange(16)[None]
                    model(input_ids=prefix)
                    reference(input_ids=prefix)
                    runtime = OptimizedDecodeRuntime(model, backend, 6,
                                                      memory=cloned_memory(model.memory), graph=False)
                    pointers = [state['raw_k'].data_ptr() for state in runtime.layers]
                    expected = []
                    for token in range(18, 24):
                        ids = torch.tensor([[token]])
                        expected.append(reference(input_ids=ids).logits[:, -1:])
                        actual = runtime.run(ids)
                        torch.testing.assert_close(actual, expected[-1], rtol=2e-5, atol=2e-6,
                                                   msg=f'{backend}, bias={bias}, token={token}')
                    runtime.reset()
                    for token, target in zip(range(18, 24), expected):
                        torch.testing.assert_close(runtime.run(torch.tensor([[token]])), target,
                                                   rtol=2e-5, atol=2e-6)
                    self.assertEqual(pointers, [state['raw_k'].data_ptr() for state in runtime.layers])

    def test_raw_path_matches_native_full_kv_with_shared_raw_weights(self):
        torch.manual_seed(42)
        model = make_model()
        raw = full_backbone(model, attn_impl='eager')
        cache = AppendCache(2, 32)
        with torch.inference_mode():
            raw(input_ids=torch.arange(16)[None], past_key_values=cache,
                cache_position=torch.arange(16), attention_mask=torch.ones(1, 16))
            runtime = OptimizedDecodeRuntime(model, 'full_kv', 6, raw_cache=cache, graph=False)
            for i, token in enumerate(range(18, 24)):
                ids = torch.tensor([[token]])
                output = raw(input_ids=ids, past_key_values=cache, cache_position=torch.tensor([16+i]),
                             attention_mask=torch.ones(1, 17+i)).last_hidden_state
                expected = model.lm_head(output).float()
                torch.testing.assert_close(runtime.run(ids), expected, rtol=2e-5, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
