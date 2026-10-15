"""Build SlimKV's BF16 FA2 forward-only decode extension."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fa2_source', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--source_root', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--cuda_arch', help='CUDA target list, e.g. 9.0, 12.0 or 9.0;12.0. '
                   'Defaults to TORCH_CUDA_ARCH_LIST or the current GPU.')
    a = p.parse_args()
    os.environ['MAX_JOBS'] = str(a.jobs)
    import torch
    requested = a.cuda_arch or os.environ.get('TORCH_CUDA_ARCH_LIST')
    if not requested:
        if not torch.cuda.is_available():
            p.error('No GPU detected. Set --cuda_arch to 9.0 or 12.0 for an offline build.')
        requested = '.'.join(map(str, torch.cuda.get_device_capability()))
    archs = requested.replace(' ', ';').split(';')
    if not archs or any(arch not in ('9.0', '12.0') for arch in archs):
        p.error('Supported CUDA targets are 9.0 (Hopper) and 12.0 (Blackwell workstation).')
    os.environ['TORCH_CUDA_ARCH_LIST'] = ';'.join(dict.fromkeys(archs))
    from torch.utils.cpp_extension import load, CUDA_HOME
    if a.jobs < 1:
        p.error('--jobs must be positive.')
    if CUDA_HOME is None:
        p.error('CUDA toolkit not found. Set CUDA_HOME to the toolkit containing nvcc.')
    nvcc_version = subprocess.check_output([str(Path(CUDA_HOME) / 'bin/nvcc'), '--version'], text=True)
    release = re.search(r'release (\d+)\.(\d+)', nvcc_version)
    if '12.0' in archs and (release is None or tuple(map(int, release.groups())) < (12, 8)):
        p.error('SM120 requires CUDA toolkit 12.8 or newer.')
    src, fa2, out = (Path(x).resolve() for x in (a.source_root, a.fa2_source, a.output_dir))
    out.mkdir(parents=True, exist_ok=True)
    files = [src / 'kernels/fa2_decode/binding.cpp', src / 'kernels/fa2_decode/forward_320.cu',
             src / 'kernels/fa2_decode/forward_dense.cu']
    if not (fa2 / 'csrc/cutlass/include/cutlass/cutlass.h').exists():
        p.error(f'CUTLASS headers are missing. Run: git -C {fa2} submodule update --init csrc/cutlass')
    module = load(name='slimkv_fa2_decode_320', sources=[str(x) for x in files],
                  extra_include_paths=[str(fa2 / 'csrc/flash_attn'), str(fa2 / 'csrc/flash_attn/src'),
                                       str(fa2 / 'csrc/cutlass/include')],
                  extra_cflags=['-O3', '-DFLASH_NAMESPACE=slimkv_flash320'],
                  extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo',
                                     '-DFLASH_NAMESPACE=slimkv_flash320',
                                     '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
                                     '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
                                     '--expt-relaxed-constexpr', '--expt-extended-lambda', '-Xptxas=-v'],
                  build_directory=str(out), verbose=True)
    record = {'torch': torch.__version__, 'cuda': torch.version.cuda,
              'cuda_arch_list': os.environ['TORCH_CUDA_ARCH_LIST'],
              'nvcc': nvcc_version,
              'fa2_commit': subprocess.check_output(['git', '-C', str(fa2), 'rev-parse', 'HEAD'], text=True).strip(),
              'cutlass_commit': subprocess.check_output(['git', '-C', str(fa2 / 'csrc/cutlass'), 'rev-parse', 'HEAD'], text=True).strip(),
              'sources_sha256': {str(x.relative_to(src)): hashlib.sha256(x.read_bytes()).hexdigest() for x in files},
              'binary': module.__file__, 'binary_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
    (out / 'build_manifest.json').write_text(json.dumps(record, indent=2))
    print('FA2_DECODE_BUILD_COMPLETE', record['binary_sha256'], flush=True)


if __name__ == '__main__':
    main()
