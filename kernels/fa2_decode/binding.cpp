// SlimKV decode-only FA2 extension. Kernel implementation is upstream FA2.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cmath>
#include "flash.h"

namespace FLASH_NAMESPACE {
void launch_decode_320(Flash_fwd_params& params, cudaStream_t stream);
}

std::vector<at::Tensor> forward320(at::Tensor q, const at::Tensor& k,
                                  const at::Tensor& v, double scale,
                                  bool causal, int splits, bool pack_q) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "Same device required");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16
                && v.scalar_type() == at::kBFloat16, "BF16 only");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "BSHD tensors required");
    TORCH_CHECK(q.size(3) == 320 && k.size(3) == 320 && v.size(3) == 320,
                "Pad transient Q/K/V to 320; persistent cache may retain actual rank");
    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1,
                "Last dimension must be contiguous");
    TORCH_CHECK(k.sizes() == v.sizes() && q.size(0) == k.size(0), "Invalid K/V shape");
    const int batch = q.size(0), original_q = q.size(1), original_heads = q.size(2);
    const int kv_heads = k.size(2), key_length = k.size(1);
    TORCH_CHECK(batch > 0 && kv_heads > 0 && original_heads % kv_heads == 0,
                "Invalid GQA head counts");
    TORCH_CHECK(original_q == 1 || original_q == 2, "Decode supports one/two queries");
    TORCH_CHECK(key_length > 0, "Handle empty branches before calling FA2");
    c10::cuda::CUDAGuard guard(q.device());
    auto* prop = at::cuda::getDeviceProperties(q.get_device());
    TORCH_CHECK(prop->major >= 8, "Ampere or newer required");
    if (original_q == 1) causal = false;
    const bool packed = pack_q && original_q == 1 && original_heads > kv_heads;
    if (packed) q = q.reshape({batch, kv_heads, original_heads / kv_heads, 320}).transpose(1, 2);
    const int queries = q.size(1), heads = q.size(2);
    auto out = at::empty_like(q);
    auto lse = at::empty({batch, heads, queries}, q.options().dtype(at::kFloat));
    FLASH_NAMESPACE::Flash_fwd_params p{};
    p.q_ptr = q.data_ptr(); p.k_ptr = k.data_ptr(); p.v_ptr = v.data_ptr();
    p.o_ptr = out.data_ptr(); p.softmax_lse_ptr = lse.data_ptr();
    p.q_batch_stride = q.stride(0); p.q_row_stride = q.stride(1); p.q_head_stride = q.stride(2);
    p.k_batch_stride = k.stride(0); p.k_row_stride = k.stride(1); p.k_head_stride = k.stride(2);
    p.v_batch_stride = v.stride(0); p.v_row_stride = v.stride(1); p.v_head_stride = v.stride(2);
    p.o_batch_stride = out.stride(0); p.o_row_stride = out.stride(1); p.o_head_stride = out.stride(2);
    p.b = batch; p.h = heads; p.h_k = kv_heads; p.h_h_k_ratio = heads / kv_heads;
    p.seqlen_q = queries; p.seqlen_k = key_length;
    p.seqlen_q_rounded = (queries + 127) / 128 * 128;
    p.seqlen_k_rounded = (key_length + 127) / 128 * 128;
    p.d = 320; p.d_rounded = 320;
    p.scale_softmax = scale; p.scale_softmax_log2 = scale * std::log2(std::exp(1.0));
    p.p_dropout = 1.f; p.p_dropout_in_uint8_t = 255; p.rp_dropout = 1.f;
    p.scale_softmax_rp_dropout = scale;
    p.window_size_left = -1; p.window_size_right = causal ? 0 : -1;
    p.is_causal = causal; p.is_bf16 = true; p.is_seqlens_k_cumulative = true;
    // Q is already reshaped into ordinary sequence/head strides. Keep the
    // upstream special KV-cache indexing flag disabled, as dense FA2 does.
    p.seqlenq_ngroups_swapped = false;
    const int n_blocks = (key_length + 63) / 64;
    if (splits == 0) {
        const int grid_heads = batch * heads;
        splits = std::min({32, n_blocks, (prop->multiProcessorCount + grid_heads - 1) / grid_heads});
        while (splits > 1 && (splits - 1) * ((n_blocks + splits - 1) / splits) >= n_blocks) --splits;
    }
    TORCH_CHECK(splits >= 1 && splits <= 32, "Support 1-32 splits");
    p.num_splits = splits;
    at::Tensor out_accum, lse_accum;
    if (splits > 1) {
        out_accum = at::empty({splits, batch, heads, queries, 320}, q.options().dtype(at::kFloat));
        lse_accum = at::empty({splits, batch, heads, queries}, q.options().dtype(at::kFloat));
        p.oaccum_ptr = out_accum.data_ptr(); p.softmax_lseaccum_ptr = lse_accum.data_ptr();
    }
    FLASH_NAMESPACE::launch_decode_320(p, at::cuda::getCurrentCUDAStream().stream());
    if (packed) {
        out = out.transpose(1, 2).reshape({batch, 1, original_heads, 320});
        lse = lse.reshape({batch, original_heads, 1});
    }
    return {out, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward320, "FA2 BF16 decode width 320",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("softmax_scale"), pybind11::arg("causal") = false,
          pybind11::arg("num_splits") = 0, pybind11::arg("pack_q") = true);
}
