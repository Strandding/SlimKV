"""Write run provenance without changing seeds, model settings or numerical behavior."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def redact(value):
    if isinstance(value, dict):
        return {k: '[REDACTED]' if any(s in k.lower() for s in ['access_token', 'api_key', 'password', 'secret'])
                else redact(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [redact(x) for x in value]
    return value


def record_run(path, config, inputs=()):
    def git(*args):
        r = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    versions = {}
    for package in ['torch', 'transformers', 'accelerate', 'datasets', 'flash-attn', 'triton', 'deepspeed', 'peft']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    sources = {}
    for directory in ['slimkv', 'main', 'tools', 'kernels', 'config', 'scripts']:
        for p in sorted((ROOT / directory).rglob('*')):
            if p.is_file() and p.suffix in ['.py', '.sh', '.slurm', '.json', '.yaml', '.cpp', '.cu']:
                sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs if Path(p).is_file()}
    hardware = {}
    try:
        import torch
        hardware['cuda_runtime'] = torch.version.cuda
        if torch.cuda.is_available():
            hardware['gpus'] = [dict(name=torch.cuda.get_device_name(i),
                                     total_memory_bytes=torch.cuda.get_device_properties(i).total_memory)
                                for i in range(torch.cuda.device_count())]
    except ImportError:
        hardware['torch_available'] = False
    safe_config = redact(config)
    # Store parsed configuration, never raw argv: flags may contain credentials.
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  python=sys.version, platform=platform.platform(), source_commit=git('rev-parse', 'HEAD'),
                  source_dirty=bool(git('status', '--porcelain')), source_sha256=sources,
                  config=safe_config, input_sha256=hashes, packages=versions, hardware=hardware,
                  distributed={k: os.environ[k] for k in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'CUDA_VISIBLE_DEVICES'] if k in os.environ},
                  checkpoint_identity='Config hashes do not identify weight bytes. Retain checkpoint or hash weights separately.')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--config', type=Path, help='JSON run configuration; credential fields are redacted.')
    p.add_argument('--inputs', type=Path, nargs='*', default=[])
    a = p.parse_args()
    record_run(a.output, json.loads(a.config.read_text()) if a.config else {}, a.inputs)


if __name__ == '__main__':
    main()
