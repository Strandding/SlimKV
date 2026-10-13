"""Validate Python imports, launcher paths and repository hygiene without GPUs."""
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    sources = []
    for folder in ['slimkv', 'main', 'tools', 'experiments/anchor_prototype']:
        sources.extend((ROOT / folder).rglob('*.py'))
    for path in sources:
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            errors.append(str(exc))
            continue
        if 'experiments' in path.parts:
            continue
        package = list(path.relative_to(ROOT).parts[:-1])
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                parts = package[:len(package) - node.level + 1]
                if node.module:
                    parts.extend(node.module.split('.'))
            elif node.module and node.module.split('.')[0] in ['slimkv', 'main', 'tools']:
                parts = node.module.split('.')
            else:
                continue
            target = ROOT.joinpath(*parts)
            if not (target.with_suffix('.py').is_file() or (target / '__init__.py').is_file()):
                errors.append(f'{path.relative_to(ROOT)}:{node.lineno}: missing module {".".join(parts)}')

    scripts = list((ROOT / 'scripts').rglob('*.sh')) + list((ROOT / 'scripts').rglob('*.slurm'))
    scripts += list((ROOT / 'config').glob('*.sh'))
    for path in scripts:
        result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
        if result.returncode:
            errors.append(result.stderr.strip())
        if path.suffix == '.slurm':
            saw_command = False
            for line in path.read_text().splitlines():
                if line.startswith('#SBATCH') and saw_command:
                    errors.append(f'{path.relative_to(ROOT)}: Slurm directive after executable command')
                if line.strip() and not line.lstrip().startswith('#'):
                    saw_command = True

    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for name in tracked:
        if name.endswith('.pyc') or name.startswith(('data/results/', 'data/fig/', 'model/')):
            errors.append(f'Generated artifact is tracked: {name}')
    print(json.dumps({'python_files': len(sources), 'shell_scripts': len(scripts), 'errors': errors}, indent=2))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
