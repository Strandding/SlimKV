"""Check that length grouping preserves all samples without padding or reordering within a group."""
import unittest
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from main.eval_longbench import EqualLengthBatchSampler
from main import eval_longbench


class LengthBatchingTest(unittest.TestCase):
    def test_every_sample_is_batched_once_at_its_original_length(self):
        lengths = [3, 7, 3, 7, 7, 2, 3]
        sampler = EqualLengthBatchSampler([list(range(n)) for n in lengths], 2)
        batches = list(sampler)
        self.assertEqual(batches, [[1, 3], [4], [0, 2], [6], [5]])
        self.assertEqual(len(sampler), len(batches))
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(len(lengths))))
        for batch in batches:
            self.assertLessEqual(len(batch), 2)
            self.assertEqual(len({lengths[i] for i in batch}), 1)

    def test_empty_input_and_invalid_batch_size(self):
        self.assertEqual(list(EqualLengthBatchSampler([], 4)), [])
        with self.assertRaises(ValueError):
            EqualLengthBatchSampler([[1]], 0)

    def test_token_budget_limits_long_batches_without_dropping_inputs(self):
        lengths = [8, 8, 8, 2, 2, 2, 2, 20]
        sampler = EqualLengthBatchSampler([list(range(n)) for n in lengths], 4, 16)
        self.assertEqual(list(sampler), [[7], [0, 1], [2], [3, 4, 5, 6]])
        with self.assertRaises(ValueError):
            EqualLengthBatchSampler([[1]], 4, 0)

    def test_reordered_predictions_are_scored_and_reloaded_by_original_index(self):
        class Tokenizer:
            eos_token_id = 2
            pad_token_id = 0

            def encode(self, text, **kwargs):
                return [9]

            def batch_decode(self, rows, **kwargs):
                return [{3: 'A', 4: 'B', 5: 'C'}[int(row[0])] for row in rows]

        class Model:
            config = SimpleNamespace(beacon_window=8)
            generation_config = SimpleNamespace(eos_token_id=2, pad_token_id=0)

            def generate(self, input_ids, **kwargs):
                return torch.cat([input_ids, input_ids[:, :1]], dim=1)

        def process(data, indices, **kwargs):
            tokens = {'a': [3, 3], 'b': [4] * 7, 'c': [5, 5]}
            ids = [tokens[x] for x in data['input']]
            return {'input_ids': ids, 'attention_mask': [[1] * len(x) for x in ids], 'index': indices}

        def collate(rows):
            self.assertEqual(len({len(row['input_ids']) for row in rows}), 1)
            return {key: torch.tensor([row[key] for row in rows])
                    for key in ['input_ids', 'attention_mask', 'index']}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = [dict(input=x, context='', answers=[answer], length=n,
                         all_classes=['A', 'B', 'C'], language='en', _id=x)
                    for x, answer, n in [('a', 'A', 2), ('b', 'B', 7), ('c', 'C', 2)]]
            (root / 'trec.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in data))
            argv = ['eval', '--model_name_or_path', 'fixture', '--eval_data', str(root),
                    '--output_dir', str(root / 'outputs'), '--result_dir', 'grouped',
                    '--tasks', 'trec', '--cpu', 'True', '--batch_size', '2',
                    '--group_by_input_length', 'True']
            with patch.object(eval_longbench, 'get_model_and_tokenizer', return_value=(Model(), Tokenizer())), \
                 patch.object(eval_longbench, 'process_longbench', process), \
                 patch.object(eval_longbench, 'DefaultDataCollator', return_value=collate), \
                 patch.dict(os.environ, {'MAP_NUM_PROC': '1', 'MAP_LOAD_FROM_CACHE_FILE': '0'}):
                for load in ['False', 'True']:
                    with patch.object(sys, 'argv', argv + ['--load_result', load]):
                        eval_longbench.main()
                    lines = (root / 'outputs/grouped/trec.json').read_text().splitlines()
                    self.assertEqual(json.loads(lines[0]), 100.)
                    self.assertEqual([json.loads(line)['index'] for line in lines[1:]], [1, 0, 2])


if __name__ == '__main__':
    unittest.main()
