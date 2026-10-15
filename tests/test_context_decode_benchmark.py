"""Validate full-cache semantics and inclusive reconstruction/RoPE accounting."""
import unittest
from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache
from tools.benchmark.benchmark_context_decode import (AppendCache, full_backbone, extend_rope_tables,
    cloned_memory, scope_times, synthetic_ids, ATTENTION, RECONSTRUCTION, ROPE)
from test_llama_reconstruction_cache import make_model


class ContextDecodeTest(unittest.TestCase):
    def test_full_cache_matches_dynamic_and_rewind_exposes_no_future_tokens(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = make_model()
        raw = full_backbone(model, attn_impl='eager')
        append, dynamic = AppendCache(2, 32), None
        with torch.inference_mode():
            for ids in [torch.arange(11)[None], torch.tensor([[12]]), torch.tensor([[13]])]:
                positions = torch.arange(append.get_seq_length(), append.get_seq_length() + ids.shape[1])
                mask = torch.ones(1, int(positions[-1]) + 1)
                left = raw(input_ids=ids, past_key_values=append, cache_position=positions,
                           attention_mask=mask).last_hidden_state
                output = raw(input_ids=ids, past_key_values=dynamic, cache_position=positions,
                             attention_mask=mask)
                right, dynamic = output.last_hidden_state, output.past_key_values
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            saved = append.key_cache[0][:, :, :11].clone()
            pointer = append.key_cache[0].data_ptr()
            append.rewind(11)
            raw(input_ids=torch.tensor([[15]]), past_key_values=append,
                cache_position=torch.tensor([11]), attention_mask=torch.ones(1, 12))
            self.assertEqual(append.get_seq_length(), 12)
            self.assertEqual(append.to_legacy_cache()[0][0].shape[2], 12)
            self.assertEqual(append.key_cache[0].data_ptr(), pointer)
            self.assertTrue(torch.equal(saved, append.key_cache[0][:, :, :11]))

    def test_memory_snapshot_is_independent_and_keeps_configuration_identity(self):
        model = make_model(beacon_decode_backend='latent')
        with torch.inference_mode():
            model(input_ids=torch.arange(17)[None])
            snapshot = cloned_memory(model.memory)
            self.assertIs(snapshot.config, model.config)
            before = snapshot.beacon_activations[0][0].clone()
            model.memory.beacon_activations[0][0].zero_()
            self.assertTrue(torch.equal(snapshot.beacon_activations[0][0], before))

    def test_nested_reconstruction_is_counted_once_and_rope_is_retained(self):
        def event(name, time, parent=None, device=torch.autograd.DeviceType.CPU):
            return SimpleNamespace(name=name, device_time_total=time, cpu_parent=parent, device_type=device,
                                   kernels=[SimpleNamespace(name='kernel', duration=time)], cpu_children=[])
        attention = event(ATTENTION, 10000)
        reconstruction = event(RECONSTRUCTION, 3000, attention)
        projection = event(RECONSTRUCTION, 2000, reconstruction)
        rope = event(ROPE, 1500, attention)
        flash = event('flash::flash_fwd_kernel', 4500, device=torch.autograd.DeviceType.CUDA)
        annotation = event(ATTENTION, 20000, device=torch.autograd.DeviceType.CUDA)
        result = scope_times([attention, reconstruction, projection, rope, flash, annotation], 1)
        self.assertEqual(result[ATTENTION], 10)
        self.assertEqual(result[RECONSTRUCTION], 3)
        self.assertEqual(result[ROPE], 1.5)
        self.assertEqual(result['fa2_core'], 4.5)

    def test_gpu_ranges_include_uncorrelated_kernels_and_exclude_annotation_spans(self):
        def event(name, start, end, device, annotation=False):
            return SimpleNamespace(name=name, cpu_parent=None, cpu_children=[], kernels=[],
                                   device_type=device, is_user_annotation=annotation,
                                   time_range=SimpleNamespace(start=start, end=end),
                                   device_time_total=end-start)
        cpu, cuda = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
        events = [event(ATTENTION, 0, 1000, cpu),
                  event(ATTENTION, 100, 500, cuda, True),
                  event(RECONSTRUCTION, 100, 300, cuda, True),
                  event(RECONSTRUCTION, 110, 250, cuda, True),
                  event(ROPE, 300, 350, cuda, True),
                  event('gemm', 150, 180, cuda),
                  event('rotary_kernel', 310, 320, cuda),
                  event('flash::flash_fwd_kernel', 400, 450, cuda),
                  event('flash::forward_annotation', 395, 460, cuda, True)]
        result = scope_times(events, 1)
        self.assertAlmostEqual(result[ATTENTION], .09)
        self.assertAlmostEqual(result[RECONSTRUCTION], .03)
        self.assertAlmostEqual(result[ROPE], .01)
        self.assertAlmostEqual(result['fa2_core'], .05)

    def test_launch_correlation_excludes_norm_inside_an_overlapping_gpu_range(self):
        cpu, cuda = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
        def event(name, start, end, device, identifier=0, annotation=False):
            return SimpleNamespace(name=name, id=identifier, thread=1, cpu_parent=None,
                                   cpu_children=[], kernels=[], device_type=device,
                                   is_user_annotation=annotation,
                                   time_range=SimpleNamespace(start=start, end=end),
                                   device_time_total=end-start)
        events = [event(ATTENTION, 0, 50, cpu),
                  event('cuLaunchKernel', 20, 21, cpu, 7),
                  event('cudaLaunchKernel', 60, 61, cpu, 8),
                  event(ATTENTION, 100, 200, cuda, annotation=True),
                  event('flash::flash_fwd_kernel', 110, 120, cuda, 7),
                  event('norm_kernel', 140, 150, cuda, 8)]
        result = scope_times(events, 1)
        self.assertAlmostEqual(result[ATTENTION], .01)
        self.assertAlmostEqual(result['fa2_core'], .01)

    def test_rope_extension_and_synthetic_input_preserve_existing_prefixes(self):
        model = make_model()
        rope = model.model.layers[0].self_attn.rotary_emb
        before = rope.cos_cached.clone()
        extend_rope_tables(model, rope.max_seq_len_cached + 5)
        self.assertTrue(torch.equal(before, rope.cos_cached[:len(before)]))
        tokenizer = SimpleNamespace(bos_token_id=1, encode=lambda text, **kwargs: [ord(c) for c in text])
        short, long = synthetic_ids(tokenizer, 128, 42), synthetic_ids(tokenizer, 256, 42)
        self.assertTrue(torch.equal(short, long[:, :128]))
        self.assertEqual(long.shape, (1, 256))


if __name__ == '__main__':
    unittest.main()
