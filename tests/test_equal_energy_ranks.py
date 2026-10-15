"""Check rank allocation against exhaustive max-min energy optimization."""
import itertools
import math
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from main.conversion.convert_kv_lowrank import (allocate_equal_energy_ranks, _prepare_equal_energy_projection,
                                   _convert_proj, main)


class EqualEnergyRankTests(unittest.TestCase):
    def test_threshold_matches_exhaustive_rank_search(self):
        spectra = [torch.tensor([9., 3., 2., 1.], dtype=torch.float64),
                   torch.tensor([4., 4., 3., 2.], dtype=torch.float64),
                   torch.tensor([5., 4., 3., 1.], dtype=torch.float64)]
        curves = []
        for s in spectra:
            energy = (s / s[0]).square().cumsum(0)
            curves.append((energy / energy[-1]).tolist())
        for multiple, budget, floor_ratio in itertools.product([1, 2], range(3, 13), [0., 0.5, 1.]):
            floor = max(multiple, math.ceil(math.ceil(floor_ratio * budget / 3) / multiple) * multiple)
            allowed = range(floor, 5, multiple)
            feasible = [r for r in itertools.product(allowed, repeat=3) if sum(r) <= budget]
            with self.subTest(multiple=multiple, budget=budget, floor_ratio=floor_ratio):
                if not feasible:
                    with self.assertRaises(ValueError):
                        allocate_equal_energy_ranks(spectra, budget, floor_ratio, multiple)
                    continue
                expected = max(min(e[r - 1] for e, r in zip(curves, ranks)) for ranks in feasible)
                result = allocate_equal_energy_ranks(spectra, budget, floor_ratio, multiple)
                self.assertAlmostEqual(result['energy_threshold'], expected, places=14)
                ranks = result['global_ranks']
                self.assertTrue(all(floor <= r <= 4 and r % multiple == 0 for r in ranks))
                self.assertEqual(sum(ranks), budget - budget % multiple)
                self.assertTrue(all(e[r - 1] >= expected for e, r in zip(curves, ranks)))

    def test_remaining_budget_uses_largest_normalized_energy_gain(self):
        result = allocate_equal_energy_ranks([[8., 4., 3., 3., 1., 1.], [8., 4., 4., 2., 0., 0.]], 5)
        self.assertEqual(result['initial_global_ranks'], [2, 2])
        self.assertEqual(result['global_ranks'], [2, 3])
        self.assertEqual(result['unused_rank_budget'], 0)

    def test_equal_spectra_have_deterministic_tie_breaking(self):
        result = allocate_equal_energy_ranks([[4., 3., 2., 1.]] * 2, 5, 0.)
        self.assertEqual(result['initial_global_ranks'], [2, 2])
        self.assertEqual(result['global_ranks'], [3, 2])

    def test_rank_alignment_never_exceeds_requested_budget(self):
        result = allocate_equal_energy_ranks([[4., 3., 2., 1.]] * 2, 7, 0.5, 2)
        self.assertEqual(result['global_ranks'], [4, 2])
        self.assertEqual(result['unused_rank_budget'], 1)

    def test_extreme_scales_do_not_change_allocation(self):
        spectra = [torch.tensor([9., 3., 2., 1.], dtype=torch.float64),
                   torch.tensor([4., 4., 3., 2.], dtype=torch.float64)]
        expected = allocate_equal_energy_ranks(spectra, 5)
        result = allocate_equal_energy_ranks([spectra[0] * 1e200, spectra[1] * 1e-200], 5)
        self.assertEqual(result['global_ranks'], expected['global_ranks'])
        self.assertAlmostEqual(result['energy_threshold'], expected['energy_threshold'])

    def test_full_budget_and_rank_deficient_spectra(self):
        result = allocate_equal_energy_ranks([[1., 0., 0., 0.]] * 2, 8)
        self.assertEqual(result['global_ranks'], [4, 4])
        self.assertEqual(result['energy_threshold'], 1.)

    def test_invalid_inputs_fail_explicitly(self):
        cases = [([], 1, 0.5, 1), ([[0., 0.]], 1, 0.5, 1),
                 ([[1., -1.]], 1, 0.5, 1), ([[1., 2.]], 1, 0.5, 1),
                 ([[float('inf'), 1.]], 1, 0.5, 1),
                 ([[1., 1.]], 3, 0.5, 1), ([[1., 1.]] * 2, 1, 0.5, 1),
                 ([[1., 1.]], 1, -0.1, 1), ([[1., 1.]], 1, float('nan'), 1),
                 ([[1., 1.]], 1, 0.5, 0), ([[1., 1.]], 1.5, 0.5, 1),
                 ([[1., 1., 1.]], 2, 0.5, 2)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                allocate_equal_energy_ranks(*args)


class AdaptiveConversionTests(unittest.TestCase):
    @staticmethod
    def models():
        from slimkv.llama3.configuration_llama import LlamaConfig
        from slimkv.llama3.modeling_llama import LlamaForCausalLM
        from slimkv.qwen2.configuration_qwen2 import Qwen2Config
        from slimkv.qwen2.modeling_qwen2 import Qwen2ForCausalLM
        for config_cls, model_cls in [(LlamaConfig, LlamaForCausalLM), (Qwen2Config, Qwen2ForCausalLM)]:
            config = config_cls(vocab_size=64, hidden_size=32, intermediate_size=48,
                                num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                                max_position_embeddings=128, beacon_param=['q', 'k', 'v'],
                                beacon_kv_compression='none', beacon_skip_rope_for_beacon_k=True,
                                beacon_window=8, beacon_stride=8, beacon_ratio=[2])
            if config_cls is LlamaConfig:
                config.rope_scaling = {'rope_type': 'llama3', 'type': 'llama3', 'factor': 8.0, 'low_freq_factor': 1.0,
                                       'high_freq_factor': 4.0, 'original_max_position_embeddings': 64}
            config._attn_implementation = 'eager'
            model = model_cls(config).eval()
            with torch.no_grad():
                patterns = [torch.tensor([10.] + [0.1] * 15), torch.ones(16),
                            torch.tensor([2.] * 8 + [0.25] * 8)]
                for i, layer in enumerate(model.model.layers):
                    for proj, pattern in [('k', patterns[i]), ('v', patterns[2 - i])]:
                        raw = getattr(layer.self_attn, f'{proj}_proj')
                        raw.weight.zero_()
                        raw.weight[:, :16].copy_(torch.diag(pattern))
                        if raw.bias is not None:
                            raw.bias.copy_(torch.arange(16) / 100.)
            yield model

    def test_raw_initialization_reuses_svd_and_preserves_backbone(self):
        for model in self.models():
            with self.subTest(model=type(model).__name__):
                layers = [layer.self_attn for layer in model.model.layers]
                backbone = {n: p.detach().clone() for n, p in model.named_parameters() if 'beacon' not in n}
                for proj in ['k', 'v']:
                    with torch.no_grad():
                        for layer in layers:
                            getattr(layer, f'beacon_{proj}_proj').weight.fill_(99.)
                    with patch('torch.linalg.svd', wraps=torch.linalg.svd) as svd:
                        dims, factors, report = _prepare_equal_energy_projection(layers, proj, 2., 0.5, 'cpu')
                        oracle = [(u[:, :r] * s[:r]) @ vh[:r]
                                  for (u, s, vh), r in zip(factors, report['global_ranks'])]
                        _convert_proj(layers, proj, dims, 'cpu', 'global', True, factors)
                        self.assertEqual(svd.call_count, len(layers))
                    self.assertEqual(report['used_rank_budget'], 24)
                    self.assertTrue(all(f is None for f in factors))
                    for layer, expected in zip(layers, oracle):
                        up = getattr(layer, f'beacon_{proj}_up_proj')
                        down = getattr(layer, f'beacon_{proj}_down_proj')
                        torch.testing.assert_close(up.weight @ down.weight, expected)
                        if up.bias is not None:
                            torch.testing.assert_close(up.bias, getattr(layer, f'{proj}_proj').bias)
                for n, p in model.named_parameters():
                    if n in backbone:
                        self.assertTrue(torch.equal(p, backbone[n]), n)

    def test_cli_saves_and_reloads_adaptive_configuration(self):
        for model in self.models():
            with self.subTest(model=type(model).__name__), tempfile.TemporaryDirectory() as output:
                source = Path(output) / 'base'
                target = Path(output) / 'converted'
                model.save_pretrained(source)
                from tokenizers import Tokenizer
                from tokenizers.models import WordLevel
                from transformers import PreTrainedTokenizerFast
                tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
                    {'[UNK]': 0, '[EOS]': 1, 'a': 2}, unk_token='[UNK]')),
                    unk_token='[UNK]', eos_token='[EOS]', pad_token='[EOS]')
                tokenizer.save_pretrained(source)
                argv = ['convert', '--model_name_or_path', str(source), '--output_dir', str(target),
                        '--dtype', 'fp32',
                        '--rank_allocation', 'equal_energy', '--feature_compression_ratio', '4',
                        '--save_dtype', 'fp32', '--save_tokenizer', 'False',
                        '--save_svd_spectra', 'False', '--save_svd_spectra_data', 'False']
                with patch('sys.argv', argv):
                    main()
                with open(target / 'rank_allocation.json') as stream:
                    report = json.load(stream)
                reloaded = type(model).from_pretrained(target).eval()
                self.assertEqual(reloaded.config.beacon_rank_allocation, report)
                self.assertGreater(len(set(reloaded.config.beacon_k_latent_head_dim_list)), 1)
                self.assertNotEqual(reloaded.config.beacon_k_latent_head_dim_list,
                                    reloaded.config.beacon_v_latent_head_dim_list)
                for proj in ['k', 'v']:
                    self.assertEqual(report['projections'][proj]['used_rank_budget'], 12)
                    self.assertEqual(getattr(reloaded.config, f'beacon_{proj}_latent_head_dim_list'),
                                     report['projections'][proj]['latent_head_dims'])
                for key, value in model.state_dict().items():
                    if 'beacon' not in key:
                        self.assertTrue(torch.equal(value, reloaded.state_dict()[key]), key)
                saved = type(model).from_pretrained(target).state_dict()
                for key, value in reloaded.state_dict().items():
                    self.assertTrue(torch.equal(value, saved[key]), key)

    def test_manual_conversion_still_uses_existing_beacon_weights(self):
        for model in self.models():
            with self.subTest(model=type(model).__name__):
                layers = [layer.self_attn for layer in model.model.layers]
                with torch.no_grad():
                    for layer in layers:
                        layer.beacon_k_proj.weight.fill_(3.)
                _convert_proj(layers, 'k', [2] * len(layers), 'cpu', 'global', False)
                for layer in layers:
                    torch.testing.assert_close(layer.beacon_k_up_proj.weight @ layer.beacon_k_down_proj.weight,
                                               torch.full_like(layer.k_proj.weight, 3.))


if __name__ == '__main__':
    unittest.main()
