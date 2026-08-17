#ifndef PTS_LIBRARY_CUH
#define PTS_LIBRARY_CUH

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef FAST_OPS_EXPORTS
        #define FAST_OPS_API __declspec(dllexport)
    #else
        #define FAST_OPS_API __declspec(dllimport)
    #endif
#else
    #define FAST_OPS_API __attribute__((visibility("default")))
#endif


struct WelfordState {
    float mean;
    float m2;
    float count;
};

// Device & Helper Primitives
__device__ WelfordState welford_merge(WelfordState a, WelfordState b);
__device__ WelfordState warp_reduce_welford(WelfordState val);
__device__ WelfordState block_reduce_welford(WelfordState val, WelfordState* smem);
__device__ float warp_reduce_sum_f32(float val);
__device__ float block_reduce_sum_f32(float val, float* smem);

// Global Kernels
__global__ void fused_layernorm_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    float* __restrict__ mean_cache,
    float* __restrict__ rstd_cache,
    int rows,
    int cols,
    float eps
);

__global__ void fused_layernorm_bwd_kernel(
    const float* __restrict__ dout,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ mean_cache,
    const float* __restrict__ rstd_cache,
    float* __restrict__ dx,
    float* __restrict__ dresidual,
    float* __restrict__ dgamma_part,
    float* __restrict__ dbeta_part,
    int rows,
    int cols
);

__global__ void fused_rmsnorm_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    float* __restrict__ out,
    float* __restrict__ rstd_cache,
    int rows,
    int cols,
    float eps
);

__global__ void fused_rmsnorm_bwd_kernel(
    const float* __restrict__ dout,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ rstd_cache,
    float* __restrict__ dx,
    float* __restrict__ dresidual,
    float* __restrict__ dgamma_part,
    int rows,
    int cols
);

__global__ void reduce_parameter_gradients_kernel(
    const float* __restrict__ dparam_part,
    float* __restrict__ dparam,
    int rows,
    int cols
);

// Host Launch & Binding Wrappers
extern "C" {

FAST_OPS_API void launch_layernorm_fwd(
    uintptr_t x,
    uintptr_t residual,
    uintptr_t gamma,
    uintptr_t beta,
    uintptr_t out,
    uintptr_t mean,
    uintptr_t rstd,
    int rows,
    int cols,
    float eps,
    cudaStream_t stream
);

FAST_OPS_API void launch_layernorm_bwd(
    uintptr_t dout,
    uintptr_t x,
    uintptr_t gamma,
    uintptr_t mean,
    uintptr_t rstd,
    uintptr_t dx,
    uintptr_t dresidual,
    uintptr_t dgamma,
    uintptr_t dbeta,
    int rows,
    int cols,
    cudaStream_t stream
);

FAST_OPS_API void launch_rmsnorm_fwd(
    uintptr_t x,
    uintptr_t residual,
    uintptr_t gamma,
    uintptr_t out,
    uintptr_t rstd,
    int rows,
    int cols,
    float eps,
    cudaStream_t stream
);

FAST_OPS_API void launch_rmsnorm_bwd(
    uintptr_t dout,
    uintptr_t x,
    uintptr_t gamma,
    uintptr_t rstd,
    uintptr_t dx,
    uintptr_t dresidual,
    uintptr_t dgamma,
    int rows,
    int cols,
    cudaStream_t stream
);

} // extern "C"

// Device & Helper Primitives
__device__ float fast_gelu_fwd_op(float x);
__device__ float fast_gelu_bwd_op(float x, float dy);
__device__ float silu_op(float x);
__device__ float silu_bwd_op(float x, float dy);

// Global Kernels
__global__ void swiglu_fwd_kernel(
    const float* __restrict__ gate_up,
    float* __restrict__ out,
    int total_elements,
    int hidden_dim
);

__global__ void swiglu_bwd_kernel(
    const float* __restrict__ dout,
    const float* __restrict__ gate_up,
    float* __restrict__ dgate_up,
    int total_elements,
    int hidden_dim
);

__global__ void geglu_fwd_kernel(
    const float* __restrict__ gate_up,
    float* __restrict__ out,
    int total_elements,
    int hidden_dim
);

__global__ void geglu_bwd_kernel(
    const float* __restrict__ dout,
    const float* __restrict__ gate_up,
    float* __restrict__ dgate_up,
    int total_elements,
    int hidden_dim
);

__global__ void fast_gelu_fwd_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int total_elements
);

__global__ void fast_gelu_bwd_kernel(
    const float* __restrict__ dout,
    const float* __restrict__ x,
    float* __restrict__ dx,
    int total_elements
);

// Host Launch & Binding Wrappers
extern "C" {

FAST_OPS_API void launch_swiglu_fwd(
    uintptr_t gate_up,
    uintptr_t out,
    int total_elements,
    int hidden_dim,
    cudaStream_t stream
);

FAST_OPS_API void launch_swiglu_bwd(
    uintptr_t dout,
    uintptr_t gate_up,
    uintptr_t dgate_up,
    int total_elements,
    int hidden_dim,
    cudaStream_t stream
);

FAST_OPS_API void launch_geglu_fwd(
    uintptr_t gate_up,
    uintptr_t out,
    int total_elements,
    int hidden_dim,
    cudaStream_t stream
);

FAST_OPS_API void launch_geglu_bwd(
    uintptr_t dout,
    uintptr_t gate_up,
    uintptr_t dgate_up,
    int total_elements,
    int hidden_dim,
    cudaStream_t stream
);

FAST_OPS_API void launch_fast_gelu_fwd(
    uintptr_t x,
    uintptr_t out,
    int total_elements,
    cudaStream_t stream
);

FAST_OPS_API void launch_fast_gelu_bwd(
    uintptr_t dout,
    uintptr_t x,
    uintptr_t dx,
    int total_elements,
    cudaStream_t stream
);

} // extern "C"

// Device & Helper Primitives
__device__ float warp_reduce_max_f32(float val);
__device__ float block_reduce_max_f32(float val, float* smem);
__device__ void online_softmax_step(float& max_val, float& sum_val, float x);

// Global Kernels
__global__ void fused_softmax_fwd_kernel(
    const float* __restrict__ logits,
    float* __restrict__ out,
    int rows,
    int cols
);

__global__ void fused_softmax_bwd_kernel(
    const float* __restrict__ dout,
    const float* __restrict__ softmax_out,
    float* __restrict__ dx,
    int rows,
    int cols
);

__global__ void fused_cross_entropy_fwd_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    float* __restrict__ lse_cache,
    int total_tokens,
    int vocab_size,
    int64_t ignore_index
);

__global__ void fused_cross_entropy_bwd_kernel(
    const float* __restrict__ dloss,
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ lse_cache,
    float* __restrict__ dlogits,
    int total_tokens,
    int vocab_size,
    int64_t ignore_index
);

// Host Launch & Binding Wrappers
extern "C" {

FAST_OPS_API void launch_softmax_fwd(
    uintptr_t logits,
    uintptr_t out,
    int rows,
    int cols,
    cudaStream_t stream
);

FAST_OPS_API void launch_softmax_bwd(
    uintptr_t dout,
    uintptr_t softmax_out,
    uintptr_t dx,
    int rows,
    int cols,
    cudaStream_t stream
);

FAST_OPS_API void launch_cross_entropy_fwd(
    uintptr_t logits,
    uintptr_t targets,
    uintptr_t losses,
    uintptr_t lse_cache,
    int total_tokens,
    int vocab_size,
    int64_t ignore_index,
    cudaStream_t stream
);

FAST_OPS_API void launch_cross_entropy_bwd(
    uintptr_t dloss,
    uintptr_t logits,
    uintptr_t targets,
    uintptr_t lse_cache,
    uintptr_t dlogits,
    int total_tokens,
    int vocab_size,
    int64_t ignore_index,
    cudaStream_t stream
);

} // extern "C"

// Device & Helper Primitives
__device__ void apply_rope_rotation_2d(
    float x1,
    float x2,
    float cos_val,
    float sin_val,
    float& out1,
    float& out2
);

// Global Kernels
__global__ void fused_rope_fwd_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
);

__global__ void fused_rope_bwd_kernel(
    const float* __restrict__ dq_out,
    const float* __restrict__ dk_out,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ dq,
    float* __restrict__ dk,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
);


extern "C" {

FAST_OPS_API void launch_rope_fwd(
    uintptr_t q,
    uintptr_t k,
    uintptr_t cos,
    uintptr_t sin,
    uintptr_t q_out,
    uintptr_t k_out,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    cudaStream_t stream
);

FAST_OPS_API void launch_rope_bwd(
    uintptr_t dq_out,
    uintptr_t dk_out,
    uintptr_t cos,
    uintptr_t sin,
    uintptr_t dq,
    uintptr_t dk,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    cudaStream_t stream
);

}

#endif //PTS_LIBRARY_CUH