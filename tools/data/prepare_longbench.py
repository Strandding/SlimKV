"""Select reproducible benchmark fixtures from local official LongBench-v1 JSONL files."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset_dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', default=['qasper', 'hotpotqa', 'multifieldqa_en', 'lcc'])
    p.add_argument('--max_samples_per_task', type=int, default=2)
    p.add_argument('--min_length', type=int, default=2500)
    p.add_argument('--max_length', type=int, default=6000)
    a = p.parse_args()
    if a.max_samples_per_task < 1 or a.min_length > a.max_length:
        p.error('Require a positive sample count and an ordered length range.')
    if len(set(a.tasks)) != len(a.tasks):
        p.error('Task names must be unique.')
    rows, sources = [], {}
    for task in a.tasks:
        source = a.dataset_dir / f'{task}.jsonl'
        data = source.read_bytes()
        sources[task] = hashlib.sha256(data).hexdigest()
        selected = []
        for index, line in enumerate(data.decode('utf-8').splitlines()):
            row = json.loads(line)
            if a.min_length <= row['length'] <= a.max_length:
                selected.append(dict(row, task=task, source_index=index))
                if len(selected) == a.max_samples_per_task:
                    break
        if len(selected) != a.max_samples_per_task:
            p.error(f'{task}: only {len(selected)} samples match the requested range.')
        rows.extend(selected)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    manifest = {'selection': vars(a), 'source_sha256': sources,
                'output_sha256': hashlib.sha256(a.output.read_bytes()).hexdigest(),
                'samples': [{'task': row['task'], 'source_index': row['source_index']} for row in rows],
                'note': 'Length filter uses dataset length, not tokenized chat prompt length; no truncation.'}
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(f'Saved {len(rows)} samples to {a.output}')


if __name__ == '__main__':
    main()
