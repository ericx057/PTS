#include "library.cuh"
#include <iostream>
#include <cuda_fp16.h>

#define cudaAssert(bool condition) if(!(condition)) throw std::invalid_argument("assert failed");

//128 bit aligned struct to load 8 halfs (16 bytes) in a single instruction
struct alignas(16) Half8 {
    half2 h2[4];
};

__device__ float warp_reduce_sum_f32(float val) {
//warpSize = 32, sum individually to remove loop overhead
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}
__device__ float block_reduce_sum_f32(float val){
    cudaAssert(warpSize == 32);
    static __shared float shared[32];

    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

//reduce in thread
    val = warp_reduce_sum_f32(val);
    if(lane == 0) shared[wid] = val;

    __syncthreads();
    if (wid == 0) {
        //complier trick, blockDim.x is const at launch
        int num_warps = (blockDim.x + 31)/ 32;
        val = (lane < num_warps) ? shared[lane] : 0.0f;
        val = warp_reduce_sum_f32(val);
    }
    return val;
}

__global__ void fused_layernorm_fwd_kernel(
    const half* __restrict__ x,
    const half* __restrict__ residual,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    half* __restrict__ out,
    float* __restrict__ mean_cache,
    float* __restrict__ rstd_cache,
    int rows,
    int cols,
    float eps
){
    cudaAssert(cols%8 == 0);
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int row = global_tid / warpSize;
    int lane = global_tid % warpSize;

    int cols8 = cols/8; 
    const Half8* x_row_h8 = reinterpret_cast<const Half8*>(x + row * cols);
    const Half8* gamma_h8 = reinterpret_cast<const Half8*>(gamma);
    const Half8* beta_h8 = reinterpret_cast<const Half8*>(beta);
    Half8* out_row_h8 = reinterpret_cast<Half8*>(out + row * cols);

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    //neat float8 optimization trick, fetch 128 bits of information allowing for high utilization
    for (int i = lane; i < cols8; i += warpSize) {
        Half8 val8 = x_row_h8[i];

        #pragma unroll
        for(int j = 0; j < 4; ++j) {
            float2 vf = __half22float2(val8.h2[j]);
            local_sum += vf.x + vf.y;
            local_sq_sum += (vf.x*vf.x) + (vf.y * vf.y);
        }
    }
    
    float block_sum = warp_reduce_sum_f32(local_sum);
    float block_sq_sum = warp_reduce_sum_f32(local_sq_sum);
    float s_mean;
    float s_invvar;
    //just compute everything at the same time because its faster than doing a second pass
    if(lane == 0){
        s_mean = block_sum/(float)cols;
        float variance = (block_sq_sum/(float)cols) - (s_mean * s_mean);
        s_invvar = rsqrtf(variance + eps);
        mean_cache[row] = s_mean;
        rstd_cache[row] = s_invvar;
    }
    
    s_mean = __shfl_sync(0xffffffff, s_mean, 0);
    s_invvar = __shfl_sync(0xffffffff, s_invvar, 0);

    for(int i = lane; i<cols8; i+= 32){
        Half8 v = x_row_h8[i];
        Half8 g = gamma_h8[i];
        Half8 b = beta_h8[i];

        //output variable
        Half8 o;

        #pragma unroll
        for(int j = 0; j < 4; ++j) {
            float2 vf = __half22float2(v.h2[j]);
            float2 gf = __half22float2(g.h2[j]);
            float2 bf = __half22float2(b.h2[j]);

            float ox = ((vf.x - s_mean)*s_invvar)*gf.x+bf.x;
            float oy = ((vf.y - s_mean)*s_invvar)*gf.y+bf.y;

            o.h2[j] = __floats2half2_rn(ox, oy);
        }

        out_row_h8[i] = o;
    }
}

//we use warp to free L1 because __shared__ is not required
__global__ void fused_layernorm_bwd_kernel_warp(
    const half* __restrict__ dout,
    const half* __restrict__ x,
    const half* __restrict__ gamma,
    const float* __restrict__ mean_cache,
    const float* __restrict__ rstd_cache,
    half* __restrict__ dx,
    half* __restrict__ dresidual,
    float* __restrict__ dgamma_part,
    float* __restrict__ dbeta_part,
    int rows,
    int cols
){
    cudaAssert(cols%8 == 0);
    
    //use same mapping
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = global_tid / warpSize;
    int lane = global_tid % warpSize;

    cudaAssert(row<rows);

    int cols8 = cols/8;
    
    //use vectorized pointers, same logic as pervious
    const Half8* dout_row_h8 = reinterpret_cast<const Half8*>(dout + row * cols);
    const Half8* x_row_h8 = reinterpret_cast<const Half8*>(x + row * cols);
    const Half8* gamma_h8 = reinterpret_cast<const Half8*>(gamma);
    
    Half8* dx_row_h8 = reinterpret_cast<Half8*>(dx + row * cols);
    Half8* dres_row_h8 = reinterpret_cast<Half8*>(dresidual + row * cols);
    
    float* dgamma_row = dgamma_part + row * cols;
    float* dbeta_row = dbeta_part + row * cols;

    //read cached stats
    float s_mean = mean_cache[row];
    float s_invvar = rstd_cache[row];

    float local_sum_ds = 0.0f;
    float local_sum_ds_xh = 0.0f;

    //setup
    for(int i = lane; i < cols8; i += warpSize){
        Half8 x_val = x_row_h8[i];
        Half8 d_val = dout_row_h8[i];
        Half8 g_val = gamma_h8[i];

        #pragma unroll //bounded loop lol
        for(int j = 0; j<4; j++) {
            float2 xf = __half22float2(x_val.h2[j]);
            float2 df = __half22float2(d_val.h2[j]);
            float2 gf = __half22float2(g_val.h2[j]);

            float x_h_x = (xf.x - s_mean) * s_invvar;
            float x_h_y = (xf.y - s_mean) * s_invvar;

            float ds_x = df.x * gf.x;
            float ds_y = df.y * gf.y;

            local_sum_ds += (ds_x + ds_y);
            local_sum_ds_xh += (ds_x * x_h_x) + (ds_y * x_h_y);
        }
    }

    //recomputation of stats and local sums
    float warp_sum_ds = warp_reduce_sum_f32(local_sum_ds);
    float warp_sum_ds_xh = warp_reduce_sum_f32(local_sum_ds_xh);

    float s_mean_ds = __shfl_sync(0xffffffff, warp_sum_ds / (float)cols, 0);
    float s_mean_ds_xh = __shfl_sync(0xffffffff, warp_sum_ds_xh / (float)cols, 0);

    for (int i = lane; i < cols8; i += warpSize) {
        Half8 x_val = x_row_h8[i];
        Half8 d_val = dout_row_h8[i];
        Half8 g_val = gamma_h8[i];

        Half8 dx_out;

        #pragma unroll
        for(int j = 0; j < 4; ++j) {
            float2 xf = __half22float2(x_val.h2[j]);
            float2 df = __half22float2(d_val.h2[j]);
            float2 gf = __half22float2(g_val.h2[j]);

            float x_h_x = (xf.x - s_mean) * s_invvar;
            float x_h_y = (xf.y - s_mean) * s_invvar;

            float dx_x = s_invvar * ((df.x * gf.x) - s_mean_ds - (x_h_x * s_mean_ds_xh));
            float dx_y = s_invvar * ((df.y * gf.y) - s_mean_ds - (x_h_y * s_mean_ds_xh));

            dx_out.h2[j] = __floats2half2_rn(dx_x, dx_y);

            int t = (i * 8) + (j * 2);
            dgamma_row[t] = df.x * x_h_x;
            dgamma_row[t+1] = df.y * x_h_y;

            dbeta_row[t] = df.x;
            dbeta_row[t+1] = df.y;
        }

        dx_row_h8[i] = dx_out;
        dres_row_h8[i] = dx_out; 
    }
}