#include "library.cuh"
#define cudaAssert(bool condition) if(!(condition)) throw std::invalid_argument("assert failed");

__device__ float warp_reduce_sum_f32(float val) {
//warpSize = 32, sum individually to remove loop overhead
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}
__device__ float block_reduce_sum_f32(float val, float* smem){
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
){
    int row = blockInd.x;
    if(row >= rows) return;

    int ThreadID = threadIdx.x;
    const float* x_row = x+row*cols;
    float* y_row = out+row*cols;

    float sum = 0.0f;
    for(int i = 0; i<cols; i++){
        sum += x_row[i];
    }
    float block_sum = block_reduce_sum_f32(sum);

    __shared__ float s_mean;

}