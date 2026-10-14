"""Build SlimKV's BF16 FA2 forward-only decode extension."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fa2_source', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--source_root', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    os.environ['MAX_JOBS'] = str(a.jobs)
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
    import torch
    from torch.utils.cpp_extension import load
    src, fa2, out = (Path(x).resolve() for x in (a.source_root, a.fa2_source, a.output_dir))
    out.mkdir(parents=True, exist_ok=True)
    files = [src / 'kernels/fa2_decode/binding.cpp', src / 'kernels/fa2_decode/forward_320.cu',
             src / 'kernels/fa2_decode/forward_dense.cu']
    assert (fa2 / 'csrc/cutlass/include/cutlass/cutlass.h').exists()
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
              'fa2_commit': subprocess.check_output(['git', '-C', str(fa2), 'rev-parse', 'HEAD'], text=True).strip(),
              'cutlass_commit': subprocess.check_output(['git', '-C', str(fa2 / 'csrc/cutlass'), 'rev-parse', 'HEAD'], text=True).strip(),
              'sources_sha256': {str(x.relative_to(src)): hashlib.sha256(x.read_bytes()).hexdigest() for x in files},
              'binary': module.__file__, 'binary_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
    (out / 'build_manifest.json').write_text(json.dumps(record, indent=2))
    print('FA2_DECODE_BUILD_COMPLETE', record['binary_sha256'], flush=True)


if __name__ == '__main__':
    main()
