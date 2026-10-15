"""Exercise fixture selection, private-field redaction and the real inference CLI."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.record_run import record_run

ROOT = Path(__file__).resolve().parents[1]


class RepositoryToolsTest(unittest.TestCase):
    def test_attention_backends_are_explicit(self):
        from slimkv import ModelArgs
        from slimkv.llama3.modeling_llama import LLAMA_ATTENTION_CLASSES
        from slimkv.qwen2.modeling_qwen2 import QWEN2_ATTENTION_CLASSES
        supported = {'eager', 'flash_attention_2'}
        self.assertEqual(set(LLAMA_ATTENTION_CLASSES), supported)
        self.assertEqual(set(QWEN2_ATTENTION_CLASSES), supported)
        self.assertEqual(ModelArgs(attn_impl=None).attn_impl, 'eager')
        for choice in supported:
            self.assertEqual(ModelArgs(attn_impl=choice).attn_impl, choice)
        with self.assertRaisesRegex(ValueError, 'eager or flash_attention_2'):
            ModelArgs(attn_impl='sdpa')

    def test_manifest_records_actual_inputs_and_redacts_nested_credentials(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = directory / 'input.json'
            source.write_text('{"seed":42}')
            output = directory / 'run.json'
            record_run(output, {'seed': 42, 'model': {'access_token': 'private-value'},
                                'items': [{'api_key': 'also-private'}]}, [source])
            saved = json.loads(output.read_text())
            self.assertEqual(saved['config']['seed'], 42)
            self.assertEqual(saved['config']['model']['access_token'], '[REDACTED]')
            self.assertNotIn('private-value', output.read_text())
            self.assertNotIn('also-private', output.read_text())
            self.assertEqual(saved['input_sha256'][str(source)], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertIn('slimkv/llama3/modeling_llama.py', saved['source_sha256'])

    def test_fixture_selection_keeps_original_indices_and_rejects_insufficient_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rows = [{'input': '', 'context': 'context', 'answers': ['answer'], 'length': n}
                    for n in [10, 3000, 6100, 5000]]
            (directory / 'qasper.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            output = directory / 'samples.jsonl'
            cmd = [sys.executable, '-m', 'tools.data.prepare_longbench', '--dataset_dir', tmp,
                   '--output', str(output), '--tasks', 'qasper']
            subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
            selected = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row['source_index'] for row in selected], [1, 3])
            self.assertEqual([row['context'] for row in selected], ['context', 'context'])
            first = output.read_bytes()
            subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
            self.assertEqual(first, output.read_bytes())
            failure = subprocess.run(cmd + ['--max_samples_per_task', '3'], cwd=ROOT, capture_output=True)
            self.assertNotEqual(failure.returncode, 0)
            self.assertEqual(first, output.read_bytes())

    def test_inference_cli_loads_real_tiny_checkpoint_and_tokenizer_offline(self):
        from test_llama_reconstruction_cache import make_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        with tempfile.TemporaryDirectory() as tmp:
            model = make_model(rope_scaling={'rope_type': 'llama3', 'type': 'llama3', 'factor': 8.0,
                                            'low_freq_factor': 1.0, 'high_freq_factor': 4.0,
                                            'original_max_position_embeddings': 128},
                               bos_token_id=2, eos_token_id=3, pad_token_id=0)
            model.save_pretrained(tmp)
            vocabulary = dict(pad=0, unk=1, bos=2, eos=3)
            vocabulary.update({f'w{i}': i for i in range(4, 64)})
            backend = Tokenizer(WordLevel(vocabulary, unk_token='unk'))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
                                                unk_token='unk', pad_token='pad',
                                                bos_token='bos', eos_token='eos')
            tokenizer.save_pretrained(tmp)
            config_before = (Path(tmp) / 'config.json').read_bytes()
            env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1')
            run = subprocess.run([sys.executable, '-m', 'main.inference.infer', '--model_name_or_path', tmp,
                                  '--device', 'cpu', '--dtype', 'fp32', '--prompt', 'w4 w5 w6',
                                  '--max_new_tokens', '3'], cwd=ROOT, env=env, text=True,
                                 capture_output=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(config_before, (Path(tmp) / 'config.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
