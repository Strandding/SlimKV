// Extends the upstream FA2 forward kernel to width 320 without compiling its
// full forward/backward matrix of dtype, width and attention variants.
// Upstream headers: Dao-AILab/flash-attention v2.8.3, BSD-3-Clause.
#include "flash_fwd_launch_template.h"
#include <ATen/cuda/CUDAContext.h>

namespace FLASH_NAMESPACE {
using Traits320 = Flash_fwd_kernel_traits<320, 64, 64, 4, false, false, cutlass::bfloat16_t>;

// SM120 has 99 KiB per block. Keep the 64-key reduction tiles and use
// fewer query rows so width-320 decode fits without changing trained ranks.
struct Traits320Compact : Flash_fwd_kernel_traits<320, 16, 64, 1, false, false, cutlass::bfloat16_t> {
    using GmemLayoutAtomOaccum = cute::Layout<cute::Shape<cute::_2, cute::_16>,
                                             cute::Stride<cute::_16, cute::_1>>;
    using GmemTiledCopyOaccum = decltype(cute::make_tiled_copy(
        cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, float>{},
        GmemLayoutAtomOaccum{}, cute::Layout<cute::Shape<cute::_1, cute::_4>>{}));
};
static_assert(Traits320Compact::kSmemSize <= 99 * 1024);

template<typename Traits, bool Causal, bool Split>
void launch_one(Flash_fwd_params& p, cudaStream_t stream) {
    auto kernel = &flash_fwd_splitkv_kernel<Traits, Causal, false, false,
                                          false, true, false, Split, false>;
    constexpr int smem = Traits::kSmemSize;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    dim3 grid((p.seqlen_q + Traits::kBlockM - 1) / Traits::kBlockM, Split ? p.num_splits : p.b,
              Split ? p.b * p.h : p.h);
    kernel<<<grid, Traits::kNThreads, smem, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int LogSplits>
void combine(Flash_fwd_params& p, cudaStream_t stream) {
    constexpr int BlockM = 8;
    dim3 grid((p.b * p.h * p.seqlen_q + BlockM - 1) / BlockM);
    flash_fwd_splitkv_combine_kernel<Traits320, BlockM, LogSplits, true>
        <<<grid, Traits320::kNThreads, 0, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename Traits>
void dispatch_decode_320(Flash_fwd_params& p, cudaStream_t stream) {
    if (p.is_causal) {
        if (p.num_splits > 1) launch_one<Traits, true, true>(p, stream);
        else launch_one<Traits, true, false>(p, stream);
    } else {
        if (p.num_splits > 1) launch_one<Traits, false, true>(p, stream);
        else launch_one<Traits, false, false>(p, stream);
    }
    if (p.num_splits > 1) {
        if (p.num_splits <= 2) combine<1>(p, stream);
        else if (p.num_splits <= 4) combine<2>(p, stream);
        else if (p.num_splits <= 8) combine<3>(p, stream);
        else if (p.num_splits <= 16) combine<4>(p, stream);
        else combine<5>(p, stream);
    }
}
void launch_decode_320(Flash_fwd_params& p, cudaStream_t stream) {
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    if (properties->sharedMemPerBlockOptin < Traits320::kSmemSize) {
        dispatch_decode_320<Traits320Compact>(p, stream);
    } else {
        dispatch_decode_320<Traits320>(p, stream);
    }
}
}  // namespace FLASH_NAMESPACE
