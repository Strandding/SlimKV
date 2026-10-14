"""Check that generation-only logit slicing preserves raw-token outputs and KV."""
import copy
import unittest

import torch

from test_llama_reconstruction_cache import make_model


class GenerationLogitsTest(unittest.TestCase):
    def test_prefill_and_decode_preserve_logits_and_memory(self):
        torch.set_num_threads(1)
        for ratio, batch, length in [(1, 1, 19), (4, 3, 19), (4, 3, 20)]:
            torch.manual_seed(42)
            reference = make_model(beacon_ratio=[ratio])
            optimized = copy.deepcopy(reference)
            head_lengths = []
            hook = optimized.lm_head.register_forward_pre_hook(
                lambda module, inputs: head_lengths.append(inputs[0].shape[1]))
            with torch.no_grad():
                inputs = torch.randint(3, 60, (batch, length))
                for step in range(6):
                    if step:
                        inputs = torch.randint(3, 60, (batch, 1))
                    full = reference(input_ids=inputs, attention_mask=torch.ones_like(inputs))
                    last = optimized(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                     num_logits_to_keep=1)
                    self.assertEqual(last.logits.shape, (batch, 1, 64))
                    torch.testing.assert_close(last.logits, full.logits[:, -1:],
                                               rtol=1e-5, atol=2e-6)
                    for name in ['sink_activations', 'raw_activations',
                                 'beacon_activations', 'window_beacon_activations']:
                        for pairs in zip(getattr(reference.memory, name),
                                         getattr(optimized.memory, name)):
                            for x, y in zip(*pairs):
                                if x is None:
                                    self.assertIsNone(y)
                                else:
                                    torch.testing.assert_close(x, y, rtol=0, atol=0)
            hook.remove()
            self.assertTrue(head_lengths and max(head_lengths) <= 2)

    def test_generate_preserves_greedy_tokens(self):
        model = make_model(pad_token_id=0, eos_token_id=2)
        inputs = torch.arange(3, 23)[None]
        full = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                              max_new_tokens=8, do_sample=False)
        model.memory.reset()
        last = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                              max_new_tokens=8, do_sample=False, num_logits_to_keep=1)
        self.assertTrue(torch.equal(full, last))

    def test_training_keeps_full_loss(self):
        reference = make_model().train()
        optimized = copy.deepcopy(reference)
        inputs = torch.arange(3, 19)[None]
        full = reference(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=inputs)
        last = optimized(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                         labels=inputs, num_logits_to_keep=1)
        torch.testing.assert_close(full.logits, last.logits, rtol=0, atol=0)
        torch.testing.assert_close(full.loss, last.loss, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
