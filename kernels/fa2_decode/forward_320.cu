// Extends the upstream FA2 forward kernel to width 320 without compiling its
// full forward/backward matrix of dtype, width and attention variants.
// Upstream headers: Dao-AILab/flash-attention v2.8.3, BSD-3-Clause.
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {
using Traits320 = Flash_fwd_kernel_traits<320, 64, 64, 4, false, false, cutlass::bfloat16_t>;

template<bool Causal, bool Split>
void launch_one(Flash_fwd_params& p, cudaStream_t stream) {
    auto kernel = &flash_fwd_splitkv_kernel<Traits320, Causal, false, false,
                                          false, true, false, Split, false>;
    constexpr int smem = Traits320::kSmemSize;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    dim3 grid((p.seqlen_q + 63) / 64, Split ? p.num_splits : p.b,
              Split ? p.b * p.h : p.h);
    kernel<<<grid, Traits320::kNThreads, smem, stream>>>(p);
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

void launch_decode_320(Flash_fwd_params& p, cudaStream_t stream) {
    if (p.is_causal) {
        if (p.num_splits > 1) launch_one<true, true>(p, stream);
        else launch_one<true, false>(p, stream);
    } else {
        if (p.num_splits > 1) launch_one<false, true>(p, stream);
        else launch_one<false, false>(p, stream);
    }
    if (p.num_splits > 1) {
        if (p.num_splits <= 2) combine<1>(p, stream);
        else if (p.num_splits <= 4) combine<2>(p, stream);
        else if (p.num_splits <= 8) combine<3>(p, stream);
        else if (p.num_splits <= 16) combine<4>(p, stream);
        else combine<5>(p, stream);
    }
}
}  // namespace FLASH_NAMESPACE
