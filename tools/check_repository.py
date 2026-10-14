"""Check syntax, local imports, documentation links and repository hygiene without GPUs."""
import ast
import json
import subprocess
import sys
import io
import re
import tokenize
from urllib.parse import unquote
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    sources = []
    for folder in ['slimkv', 'main', 'tools', 'tests', 'experiments/anchor_prototype']:
        sources.extend((ROOT / folder).rglob('*.py'))
    for path in sources:
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            errors.append(str(exc))
            continue
        for token in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
            if token.type == tokenize.COMMENT and re.search(r'[\u4e00-\u9fff]', token.string):
                errors.append(f'{path.relative_to(ROOT)}:{token.start[0]}: translate comment to English')
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc and re.search(r'[\u4e00-\u9fff]', doc):
                    errors.append(f'{path.relative_to(ROOT)}: translate docstring to English')
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
    scripts += list((ROOT / 'experiments/anchor_prototype/scripts').glob('*.sh'))
    for path in scripts:
        result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
        if result.returncode:
            errors.append(result.stderr.strip())
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith('#') and re.search(r'[\u4e00-\u9fff]', line):
                errors.append(f'{path.relative_to(ROOT)}:{i}: translate comment to English')
        if path.suffix == '.slurm':
            saw_command = False
            for line in path.read_text().splitlines():
                if line.startswith('#SBATCH') and saw_command:
                    errors.append(f'{path.relative_to(ROOT)}: Slurm directive after executable command')
                if line.strip() and not line.lstrip().startswith('#'):
                    saw_command = True

    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for name in tracked:
        if name.endswith(('.pyc', '.pt', '.pth', '.safetensors', '.so', '.o', '.sqlite', '.nsys-rep', '.ncu-rep')) or name.startswith(('data/results/', 'data/fig/', 'data/datasets/', 'model/')):
            errors.append(f'Generated artifact is tracked: {name}')
    docs = list(ROOT.glob('*.md')) + list((ROOT / 'docs').rglob('*.md'))
    docs += list((ROOT / 'experiments').rglob('README*.md'))
    link_count = 0
    for path in docs:
        prose = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', prose):
            target = target.strip().strip('<>')
            if re.match(r'(https?://|mailto:)', target):
                continue
            filename, _, anchor = unquote(target).partition('#')
            dest = (path.parent / filename).resolve() if filename else path
            link_count += 1
            if not dest.exists() or not dest.is_relative_to(ROOT):
                errors.append(f'{path.relative_to(ROOT)}: broken local link {target}')
            elif anchor and dest.suffix == '.md':
                headings = re.findall(r'^#+\s+(.+)$', dest.read_text(), flags=re.M)
                slugs = [re.sub(r'[^\w\- ]', '', h.lower().replace('`', '')).replace(' ', '-') for h in headings]
                if anchor not in slugs:
                    errors.append(f'{path.relative_to(ROOT)}: missing heading {target}')
    secret = re.compile(r'\b(?:hf_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,}|sk-[A-Za-z0-9_-]{24,}|AKIA[0-9A-Z]{16})\b')
    for path in sources + scripts:
        if secret.search(path.read_text()):
            errors.append(f'{path.relative_to(ROOT)}: possible embedded credential (value withheld)')
    print(json.dumps({'python_files': len(sources), 'shell_scripts': len(scripts),
                      'markdown_files': len(docs), 'local_links': link_count, 'errors': errors}, indent=2))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
