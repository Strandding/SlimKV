// Preserve the upstream dense FA2 tile choices with device cache lengths.
// Upstream headers: Dao-AILab/flash-attention v2.8.3, BSD-3-Clause.
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {
using Traits128 = Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t>;
using Split128 = Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::bfloat16_t>;
using Traits256 = Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, cutlass::bfloat16_t>;

template<typename Traits, bool EvenK, int LogSplits>
void combine_dense(Flash_fwd_params& p, cudaStream_t stream) {
    constexpr int BlockM = 4;
    dim3 grid((p.b * p.h * p.seqlen_q + BlockM - 1) / BlockM);
    flash_fwd_splitkv_combine_kernel<Traits, BlockM, LogSplits, EvenK>
        <<<grid, Traits::kNThreads, 0, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename Traits, bool Causal, bool EvenK>
void launch_split_dense(Flash_fwd_params& p, cudaStream_t stream) {
    auto kernel = &flash_fwd_splitkv_kernel<Traits, Causal, false, false, false,
                                           EvenK, false, true, false>;
    constexpr int smem = Traits::kSmemSize;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    dim3 grid((p.seqlen_q + Traits::kBlockM - 1) / Traits::kBlockM, p.num_splits, p.b * p.h);
    kernel<<<grid, Traits::kNThreads, smem, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (p.num_splits <= 2) combine_dense<Traits, EvenK, 1>(p, stream);
    else if (p.num_splits <= 4) combine_dense<Traits, EvenK, 2>(p, stream);
    else if (p.num_splits <= 8) combine_dense<Traits, EvenK, 3>(p, stream);
    else if (p.num_splits <= 16) combine_dense<Traits, EvenK, 4>(p, stream);
    else if (p.num_splits <= 32) combine_dense<Traits, EvenK, 5>(p, stream);
    else if (p.num_splits <= 64) combine_dense<Traits, EvenK, 6>(p, stream);
    else combine_dense<Traits, EvenK, 7>(p, stream);
}

template<typename Traits, bool Causal, bool EvenK>
void launch_one_dense(Flash_fwd_params& p, cudaStream_t stream) {
    auto kernel = &flash_fwd_kernel<Traits, false, Causal, false, false, false,
                                   EvenK, false, false>;
    constexpr int smem = Traits::kSmemSize;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    dim3 grid((p.seqlen_q + Traits::kBlockM - 1) / Traits::kBlockM, p.b, p.h);
    kernel<<<grid, Traits::kNThreads, smem, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<bool Causal>
void launch_dense(Flash_fwd_params& p, cudaStream_t stream) {
    if (p.num_splits > 1) {
        if (p.d == 128) launch_split_dense<Split128, Causal, true>(p, stream);
        else if (p.d == 256) launch_split_dense<Traits256, Causal, true>(p, stream);
        else launch_split_dense<Traits256, Causal, false>(p, stream);
    } else if (p.d == 128) launch_one_dense<Traits128, Causal, true>(p, stream);
    else if (p.d == 256) launch_one_dense<Traits256, Causal, true>(p, stream);
    else launch_one_dense<Traits256, Causal, false>(p, stream);
}

void launch_decode_dense(Flash_fwd_params& p, cudaStream_t stream) {
    if (p.is_causal) launch_dense<true>(p, stream);
    else launch_dense<false>(p, stream);
}
}  // namespace FLASH_NAMESPACE
