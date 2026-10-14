"""Check that length grouping preserves all samples without padding or reordering within a group."""
import unittest
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from main.eval_longbench import EqualLengthBatchSampler
from main import eval_longbench
from main.longbench_utils import ENGLISH_TASKS
from tools.download_longbench import extract_tasks


class LengthBatchingTest(unittest.TestCase):
    def test_english_preset_and_selected_archive_extraction(self):
        self.assertEqual(len(ENGLISH_TASKS), 15)
        self.assertIn('passage_retrieval_en', ENGLISH_TASKS)
        self.assertIn('repobench-p', ENGLISH_TASKS)
        self.assertNotIn('passage_count', ENGLISH_TASKS)
        self.assertFalse(any(name.endswith('_zh') for name in ENGLISH_TASKS))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'data.zip'
            with zipfile.ZipFile(archive, 'w') as zipped:
                zipped.writestr('data/qasper.jsonl', '{"input": "fixture"}\n')
                zipped.writestr('data/qasper_e.jsonl', '{"input": "other"}\n')
                zipped.writestr('../unexpected.jsonl', '{}\n')
            result = extract_tasks(archive, root / 'output', ['qasper'])
            self.assertEqual(result['tasks']['qasper']['num_samples'], 1)
            self.assertEqual(sorted(p.name for p in (root / 'output').iterdir()), ['qasper.jsonl'])
            self.assertFalse((root / 'unexpected.jsonl').exists())
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                extract_tasks(archive, root / 'output', ['hotpotqa'])
            (root / 'output/qasper.jsonl').write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'different contents'):
                extract_tasks(archive, root / 'output', ['qasper'])

    def test_final_chat_token_limit_keeps_exact_head_and_tail(self):
        class Tokenizer:
            def __call__(self, text):
                ids = list(range(10))
                return {'input_ids': ids, 'attention_mask': [1] * len(ids)}

            def encode(self, *args, **kwargs):
                raise AssertionError('Final-token truncation must not decode or re-encode the prompt.')

        chat = SimpleNamespace(encoded={'input_ids': list(range(20)), 'attention_mask': [1] * 20})
        with patch.object(eval_longbench, 'apply_chat_template', return_value=chat):
            for task, original in [('qasper', list(range(20))), ('lcc', list(range(10)))]:
                for limit in [1, 5, 6, 25]:
                    result = eval_longbench.process_longbench(
                        {'input': ['question'], 'context': ['context']}, [7],
                        Tokenizer(), 'llama-3', task, max_length=limit, truncate_after_chat=True)
                    expected = original
                    if len(original) > limit:
                        expected = original[:limit // 2] + original[-(limit - limit // 2):]
                    self.assertEqual(result['input_ids'], [expected])
                    self.assertEqual(result['attention_mask'], [[1] * len(expected)])
                    self.assertEqual(result['index'], [7])
                result = eval_longbench.process_longbench(
                    {'input': ['question'], 'context': ['context']}, [7],
                    Tokenizer(), 'llama-3', task, max_length=5,
                    truncate_after_chat=True, truncate_from_middle=False)
                self.assertEqual(result['input_ids'], [original[-5:]])

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
            generation_config = SimpleNamespace(eos_token_id=[2], pad_token_id=0)

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
                original_timing = None
                for load in ['False', 'True']:
                    with patch.object(sys, 'argv', argv + ['--load_result', load]):
                        eval_longbench.main()
                    self.assertEqual(Model.generation_config.eos_token_id, [2])
                    lines = (root / 'outputs/grouped/trec.json').read_text().splitlines()
                    self.assertEqual(json.loads(lines[0]), 100.)
                    self.assertEqual([json.loads(line)['index'] for line in lines[1:]], [1, 0, 2])
                    report = root / 'outputs/grouped'
                    timing = json.loads((report / 'trec.timing.json').read_text())
                    self.assertEqual(timing['input_tokens'], 11)
                    self.assertEqual(timing['output_tokens'], 3)
                    self.assertEqual(timing['batch_size_counts'], {'1': 1, '2': 1})
                    self.assertEqual(json.loads((report / 'scores.json').read_text())['avg'], 100.)
                    self.assertEqual(json.loads((report / 'progress.json').read_text())['status'], 'complete')
                    self.assertIn('trec,100.0,3,', (report / 'task_timings.csv').read_text())
                    if original_timing is None:
                        original_timing = timing
                    else:
                        self.assertEqual(timing, original_timing)
                with patch.object(sys, 'argv', argv + ['--load_result', 'True', '--max_length', '6']):
                    with self.assertRaisesRegex(ValueError, 'configuration differs'):
                        eval_longbench.main()
                result = root / 'outputs/grouped/trec.json'
                result.write_text('\n'.join(result.read_text().splitlines()[:-1]) + '\n')
                with patch.object(sys, 'argv', argv + ['--load_result', 'True']):
                    with self.assertRaisesRegex(ValueError, 'exactly once'):
                        eval_longbench.main()


if __name__ == '__main__':
    unittest.main()
