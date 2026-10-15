"""Check append-only cache algebra independently of CUDA Graph execution."""
import copy
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_llama_reconstruction_cache import make_model
from slimkv.llama3.latent_decode import eager_forward
from slimkv.llama3.static_decode import StaticDecode
from slimkv.modeling_utils import ModelOutput


def cache_attention(q, k, v, cache_seqlens, softmax_scale, causal, **kwargs):
    length = int(cache_seqlens[0])
    return eager_forward(q, k[:, :length], v[:, :length], softmax_scale, causal)


def varlen_attention(q, k, v, cuq, cuk, maxq, maxk, dropout, scale, causal, *args):
    length = int(cuk[-1])
    out, lse = eager_forward(q[None], k[None, :length], v[None, :length], scale, causal)
    return out[0], lse[0], None, None


class StaticDecodeTest(unittest.TestCase):
    def test_matches_reference_across_insertions_windows_reset_and_chunk_resume(self):
        torch.set_num_threads(1)
        for bias, shared, output_projection in [(False, False, False), (True, False, True), (False, True, False)]:
            torch.manual_seed(42)
            settings = dict(beacon_decode_backend='latent', attention_bias=bias,
                            beacon_shared_kv_down_proj=shared)
            if shared:
                settings['beacon_v_latent_head_dim_list'] = [2, 3]
            if output_projection:
                settings['beacon_param'] = ['q', 'k', 'v', 'o']
            model = make_model(**settings)
            reference = copy.deepcopy(model)
            mock_flash = SimpleNamespace(flash_attn_with_kvcache=cache_attention)
            with torch.inference_mode(), patch.dict(sys.modules, {
                    'flash_attn': mock_flash,
                    'flash_attn.flash_attn_interface': SimpleNamespace(_flash_attn_varlen_forward=varlen_attention)}), \
                    patch('slimkv.llama3.static_decode.cache_extension', return_value=SimpleNamespace(
                        forward=lambda q, k, v, scale, causal, splits, pack, lengths:
                        cache_attention(q, k, v, lengths, scale, causal))):
                engine = StaticDecode(model, 32)
                for state, layer in zip(engine.layers, model.model.layers):
                    for kind in ('k', 'v'):
                        latent, padded = state['latent_' + kind], state['padded_' + kind]
                        self.assertEqual(latent.untyped_storage().data_ptr(),
                                         padded.untyped_storage().data_ptr())
                        rank = getattr(layer.self_attn, 'beacon_' + kind + '_up_proj').in_features
                        self.assertEqual(latent.shape[-1], rank)
                        self.assertTrue(torch.equal(padded[..., rank:],
                                                    torch.zeros_like(padded[..., rank:])))
                for prefix in [17, 3]:
                    model.memory.reset()
                    reference.memory.reset()
                    ids = torch.arange(prefix)[None]
                    model(input_ids=ids)
                    reference(input_ids=ids)
                    initialized = False
                    for token in range(18, 47):
                        ids = torch.tensor([[token]])
                        expected = reference(input_ids=ids).logits
                        model.memory.prepare(ids, torch.ones_like(ids), None)
                        input_ids, mask, positions, past, labels = model.memory.step()
                        pk = past[0][0]
                        count = lambda x: 0 if x is None else x.shape[2]
                        latent = count(pk.history) + count(pk.window)
                        if not initialized:
                            engine.initialize(past)
                            initialized = True
                        size = input_ids.shape[1] - 1
                        raw = count(pk.sink) + count(pk.raw)
                        engine.raw_index.fill_(raw)
                        engine.latent_index.fill_(latent)
                        engine.raw_lengths.fill_(raw + 1)
                        engine.latent_lengths.fill_(latent + size)
                        model.memory._fast_window_roles = model.memory._beacon_indices != 0
                        model.memory._fast_window_raw_indices = (model.memory._beacon_indices == 0).nonzero().flatten()
                        # The CUDA implementation switches to static decode only after
                        # at least one Beacon exists; this fixture can also check zero.
                        logits = engine.forward(input_ids, positions[:, -input_ids.shape[1]:],
                                                refresh_sink=pk.history is None)
                        engine.commit(past, size)
                        actual = model.memory.output(ModelOutput(logits=logits)).logits
                        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6,
                                                   msg=f'prefix={prefix}, token={token}, bias={bias}, shared={shared}')
                        self.assertEqual(model.memory.get_memory_size(), reference.memory.get_memory_size())
                        self.assertEqual(model.memory.cache_bytes(), reference.memory.cache_bytes())
                    # A later multi-token chunk consumes published real-rank views.
                    ids = torch.tensor([[48, 49, 50]])
                    torch.testing.assert_close(model(input_ids=ids).logits, reference(input_ids=ids).logits,
                                               rtol=2e-5, atol=3e-6)


if __name__ == '__main__':
    unittest.main()
