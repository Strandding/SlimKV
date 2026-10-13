"""Exercise the real Llama model with full and latent reconstruction caches."""
import copy
import gc
import tempfile
import unittest
import weakref

import torch

from slimkv.llama3.configuration_llama import LlamaConfig
from slimkv.llama3.memory import LlamaMemory, ReconstructionCache
from slimkv.llama3.modeling_llama import LlamaForCausalLM
from slimkv.memory import Memory


def make_model(**kwargs):
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, beacon_param=['q', 'k', 'v'],
        beacon_window=8, beacon_stride=8, beacon_ratio=[4],
        beacon_ratio_mix='sequence', beacon_sink_size=1,
        beacon_skip_rope_for_beacon_k=True,
        beacon_k_compression='lowrank_svd', beacon_v_compression='lowrank_svd',
        beacon_k_latent_head_dim_list=[2, 3], beacon_v_latent_head_dim_list=[3, 2],
        torch_dtype=torch.float32,
    )
    backend = kwargs.pop('attn_impl', 'eager')
    for k, v in kwargs.items():
        setattr(config, k, v)
    config._attn_implementation = backend
    model = LlamaForCausalLM(config)
    model.memory = LlamaMemory(config)
    with torch.no_grad():
        # Both raw and Beacon paths, including up-projection biases, are nonzero.
        for p in model.parameters():
            p.normal_(0, 0.1)
    return model.eval()


class ReconstructionCacheTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def paired_models(self, **kwargs):
        latent = make_model(**kwargs)
        full = copy.deepcopy(latent)
        full.config.beacon_cache_mode = 'full'
        return latent, full

    def check_step(self, latent, full, ids):
        x = latent(input_ids=ids, attention_mask=torch.ones_like(ids))
        y = full(input_ids=ids, attention_mask=torch.ones_like(ids))
        torch.testing.assert_close(x.logits, y.logits, rtol=1e-5, atol=2e-6)
        self.assertEqual(latent.memory.get_memory_size(), full.memory.get_memory_size())
        for layer, (k, v, _, _) in enumerate(x.past_key_values):
            self.assertIsInstance(k, ReconstructionCache)
            self.assertIsInstance(v, ReconstructionCache)
            # The exported cache only reads persistent storage, with no full K/V field.
            self.assertIs(k.history, latent.memory.beacon_activations[layer][0])
            self.assertIs(v.window, latent.memory.window_beacon_activations[layer][1])
            attn = latent.model.layers[layer].self_attn
            expected = full.memory._get_layer_past_key_value(layer, 0, full.memory._beacon_indices, False)
            torch.testing.assert_close(k.materialize(attn), expected[0], rtol=1e-5, atol=2e-6)
            torch.testing.assert_close(v.materialize(attn), expected[1], rtol=1e-5, atol=2e-6)
        return x, y

    def test_prefill_decode_and_window_completion(self):
        for backend, bias, headwise, group, shared in [
            ('eager', False, False, 1, False),
            ('eager', True, False, 1, False),
            ('eager', True, True, 1, False),
            ('eager', True, True, 2, False),
            ('eager', False, False, 1, True),
            ('sdpa', False, False, 1, False),
        ]:
            with self.subTest(backend=backend, bias=bias, headwise=headwise, group=group, shared=shared):
                settings = dict(attn_impl=backend, attention_bias=bias,
                                beacon_headwise_up_proj=headwise,
                                beacon_up_proj_group_size=group,
                                beacon_shared_kv_down_proj=shared)
                if shared:
                    settings['beacon_v_latent_head_dim_list'] = [2, 3]
                latent, full = self.paired_models(**settings)
                with torch.no_grad():
                    self.check_step(latent, full, torch.arange(19)[None])
                    for token in range(20, 30):
                        self.check_step(latent, full, torch.tensor([[token]]))
                    self.assertGreater(latent.memory.cache_bytes()['window_beacon'], 0)
                    self.assertLess(sum(latent.memory.cache_bytes().values()),
                                    sum(full.memory.cache_bytes().values()))

    def test_batch_and_rope_enabled(self):
        latent, full = self.paired_models(beacon_skip_rope_for_beacon_k=False)
        with torch.no_grad():
            self.check_step(latent, full, torch.arange(38).reshape(2, 19))
            self.check_step(latent, full, torch.tensor([[40], [41]]))

    def test_persistent_storage_has_no_full_projection_backing(self):
        latent, full = self.paired_models()
        with torch.no_grad():
            output, _ = self.check_step(latent, full, torch.arange(20)[None])
            for group in [latent.memory.sink_activations, latent.memory.raw_activations,
                          latent.memory.beacon_activations, latent.memory.window_beacon_activations]:
                for pair in group:
                    for tensor in pair:
                        if tensor is not None:
                            self.assertEqual(tensor.untyped_storage().nbytes(),
                                             tensor.numel() * tensor.element_size())
            for idx, (key, value) in enumerate(latent.memory.beacon_activations):
                self.assertEqual(key.shape[-1], latent.config.beacon_k_latent_head_dim_list[idx])
                self.assertEqual(value.shape[-1], latent.config.beacon_v_latent_head_dim_list[idx])
            read = output.past_key_values[0][0]
            materialized = read.materialize(latent.model.layers[0].self_attn)
            ref = weakref.ref(materialized)
            del materialized
            gc.collect()
            self.assertIsNone(ref())

    def test_reset_and_generate(self):
        latent, full = self.paired_models()
        ids = torch.arange(19)[None]
        with torch.no_grad():
            a = latent.generate(ids, max_new_tokens=10, do_sample=False, eos_token_id=[], pad_token_id=0)
            b = full.generate(ids, max_new_tokens=10, do_sample=False, eos_token_id=[], pad_token_id=0)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            latent.memory.reset()
            full.memory.reset()
            self.assertEqual(sum(latent.memory.cache_bytes().values()), 0)
            a2 = latent.generate(ids, max_new_tokens=10, do_sample=False, eos_token_id=[], pad_token_id=0)
            torch.testing.assert_close(a, a2, rtol=0, atol=0)

    def test_training_matches_original_memory_and_gradients(self):
        model = make_model(beacon_cache_mode='latent').train()
        reference = copy.deepcopy(model)
        reference.memory = Memory(reference.config).train()
        ids = torch.arange(19)[None]
        a = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
        b = reference(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
        self.assertFalse(model.memory.latent_cache_active)
        torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
        torch.testing.assert_close(a.loss, b.loss, rtol=0, atol=0)
        a.loss.backward()
        b.loss.backward()
        for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
            self.assertEqual(p.grad is None, q.grad is None, name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)

    def test_auto_fallback_and_explicit_unsupported_mode(self):
        for settings in [dict(beacon_ratio=[0, 4]),
                         dict(beacon_chunk_fixed_rope_for_beacon_k=True),
                         dict(beacon_enhance_k_pos_residual=True)]:
            with self.subTest(settings=settings):
                memory = LlamaMemory(make_model(**settings).config).eval()
                memory._select_cache_mode()
                self.assertFalse(memory.latent_cache_active)
                memory.reset()
                memory.config.beacon_cache_mode = 'latent'
                with self.assertRaises(ValueError):
                    memory._select_cache_mode()

    def test_checkpoint_loading_preserves_weights(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path)
            loaded = LlamaForCausalLM.from_pretrained(path, attn_implementation='eager')
            self.assertIsInstance(loaded.memory, LlamaMemory)
            self.assertEqual(loaded.training, loaded.memory.training)
            self.assertEqual(model.state_dict().keys(), loaded.state_dict().keys())
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, loaded.state_dict()[k], rtol=0, atol=0, msg=k)
            loaded.eval()
            with torch.no_grad():
                output = loaded(input_ids=torch.arange(19)[None])
            self.assertTrue(loaded.memory.latent_cache_active)
            self.assertIsInstance(output.past_key_values[0][0], ReconstructionCache)

    def test_mode_change_requires_reset(self):
        model = make_model()
        with torch.no_grad():
            model(input_ids=torch.arange(19)[None])
        model.train()
        with self.assertRaises(RuntimeError):
            model.memory._select_cache_mode()
        model.memory.reset()
        model.memory._select_cache_mode()
        self.assertFalse(model.memory.latent_cache_active)

    def test_short_prefill_then_decode(self):
        latent, full = self.paired_models()
        with torch.no_grad():
            self.check_step(latent, full, torch.arange(3)[None])
            for token in range(4, 16):
                self.check_step(latent, full, torch.tensor([[token]]))

    def test_native_raw_and_append_fallback(self):
        for native in [False, True]:
            with self.subTest(native=native):
                model = make_model(beacon_pos='append')
                model._enable_beacon = not native
                reference = copy.deepcopy(model)
                reference.memory = Memory(reference.config).eval()
                with torch.no_grad():
                    for ids in [torch.arange(19)[None], torch.tensor([[20]])]:
                        positions = torch.arange(ids.shape[1])[None]
                        a = model(input_ids=ids, position_ids=positions)
                        b = reference(input_ids=ids, position_ids=positions)
                        torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
                self.assertFalse(model.memory.latent_cache_active)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
