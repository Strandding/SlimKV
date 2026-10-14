"""Download or extract the official LongBench-v1 English/code task files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from main.longbench_utils import ENGLISH_TASKS, DATASET2PROMPT

DATASET_ID = 'zai-org/LongBench'
DATASET_REVISION = '5e628be450b7e67fb7ae6e201bd6d8f7056f7672'


def extract_tasks(archive, output_dir, tasks):
    """Extract only selected v1 JSONL members, without trusting archive paths."""
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in DATASET2PROMPT for task in tasks):
        raise ValueError('Specify unique, supported LongBench-v1 tasks.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'archive_sha256': hashlib.sha256(Path(archive).read_bytes()).hexdigest(), 'tasks': {}}
    with zipfile.ZipFile(archive) as zipped:
        members = {}
        for task in tasks:
            matches = [name for name in zipped.namelist()
                       if name in [f'{task}.jsonl', f'data/{task}.jsonl']]
            if len(matches) != 1:
                raise ValueError(f'{task}: expected exactly one v1 JSONL file in the archive.')
            members[task] = matches[0]
        for task, member in members.items():
            content = zipped.read(member)
            count = len(content.decode('utf-8').splitlines())
            target = output_dir / f'{task}.jsonl'
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f'{target} already exists with different contents.')
            target.write_bytes(content)
            manifest['tasks'][task] = {'num_samples': count, 'sha256': hashlib.sha256(content).hexdigest()}
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, default=Path('data/datasets/longbench'))
    parser.add_argument('--archive', type=Path, help='Use an existing official data.zip without downloading.')
    parser.add_argument('--revision', default=DATASET_REVISION)
    parser.add_argument('--tasks', nargs='+', default=['english'])
    args = parser.parse_args()
    tasks = ENGLISH_TASKS if args.tasks == ['english'] else list(DATASET2PROMPT) if args.tasks == ['all'] else args.tasks
    archive = args.archive
    if archive is None:
        from huggingface_hub import hf_hub_download
        archive = Path(hf_hub_download(DATASET_ID, 'data.zip', repo_type='dataset', revision=args.revision))
    manifest = extract_tasks(archive, args.output_dir, tasks)
    manifest['source'] = ({'archive': str(archive.resolve())} if args.archive is not None
                          else {'dataset_id': DATASET_ID, 'revision': args.revision, 'filename': 'data.zip'})
    (args.output_dir / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Saved {sum(row["num_samples"] for row in manifest["tasks"].values())} samples '
          f'across {len(tasks)} tasks to {args.output_dir}')


if __name__ == '__main__':
    main()
