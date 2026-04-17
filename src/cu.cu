#include "cu.cuh"

#include <cfloat>
#include "errorUtils.hpp"


template<int WARPS_PER_BLOCK>
__global__ void matmul_kernel_fp32(float *xout, const float *x, const float *w, int n, int d) {
    extern __shared__ unsigned char shm[];
    float *warp_sums_shm = (float*)shm;
    // [0, 1, ..., WARPS_PER_BLOCK-1]

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x;

    if (row >= n) return;

    const float *w_row = w + row * d;
    float sum = 0.0f;

    int f4d = d / 4;
    const float4 *w_row_f4 = reinterpret_cast<const float4*>(w_row);
    const float4 *x_f4 = reinterpret_cast<const float4*>(x);
    for (int col = threadIdx.x; col < f4d; col += blockDim.x) {
        float4 w4 = w_row_f4[col];
        float4 x4 = x_f4[col];

        sum += (w4.x * x4.x +
                w4.y * x4.y +
                w4.z * x4.z +
                w4.w * x4.w);
    }

    // warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    warp_sums_shm[warp_id] = sum; // store warp sum in shared memory
    __syncthreads();

    sum = (threadIdx.x < WARPS_PER_BLOCK) ? warp_sums_shm[threadIdx.x] : 0.0f;

    if (threadIdx.x < 32) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    if (threadIdx.x == 0) {
        xout[row] = sum;
    }
}

__global__ void rmsnorm_kernel_fp32(float *xout, const float *x, const float *weight, int d, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d) {
        float mean_sq = 0.0f;
        for (int i = 0; i < d; ++i) {
            mean_sq += x[i] * x[i];
        }
        mean_sq /= d;

        float rms = sqrtf(mean_sq + eps);

        xout[idx] = (x[idx] / rms) * weight[idx];
    }
}

__global__ void rope_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    int pos,
    float theta,
    int dim,
    int kv_dim,
    int head_dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int i = tid * 2;

    if (i >= dim) return;

    int head_i = i % head_dim;

    float freq = 1.0f / powf(theta, static_cast<float>(head_i) / static_cast<float>(head_dim));
    float val = freq * static_cast<float>(pos);

    float fci, fcr;
    __sincosf(val, &fci, &fcr);

    float q0 = q[i];
    float q1 = q[i + 1];

    q[i]     = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;

    if (i < kv_dim) {
        float k0 = static_cast<float>(k[i]);
        float k1 = static_cast<float>(k[i + 1]);

        k[i]     = k0 * fcr - k1 * fci;
        k[i + 1] = k0 * fci + k1 * fcr;
    }
}

__global__ void silu_kernel(float *x, const float *x2, int d) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < d) {
        x[tid] = x[tid] / (1.0f + expf(-x[tid]));
        x[tid] *= x2[tid];
    }
}

__global__ void residual_kernel(float *x, const float *res, int d) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < d) {
        x[tid] += res[tid];
    }
}

__device__ inline float block_reduce_sum(float val, float* shared_mem) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int n_warps = blockDim.x / 32;

    for (int i = 16; i > 0; i >>= 1) val += __shfl_down_sync(0xffffffff, val, i);
    if (lane_id == 0) shared_mem[warp_id] = val;
    __syncthreads();

    float sum = (threadIdx.x < n_warps) ? shared_mem[lane_id] : 0.0f;
    if (warp_id == 0) {
        for (int i = 16; i > 0; i >>= 1) sum += __shfl_down_sync(0xffffffff, sum, i);
    }
    if (threadIdx.x == 0) shared_mem[0] = sum;
    __syncthreads();
    return shared_mem[0];
}

__device__ inline float block_reduce_max(float val, float* shared_mem) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int n_warps = blockDim.x / 32;

    for (int i = 16; i > 0; i >>= 1) val = fmaxf(val, __shfl_down_sync(0xffffffff, val, i));
    if (lane_id == 0) shared_mem[warp_id] = val;
    __syncthreads();

    float max_val = (threadIdx.x < n_warps) ? shared_mem[lane_id] : -FLT_MAX;
    if (warp_id == 0) {
        for (int i = 16; i > 0; i >>= 1) max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, i));
    }
    if (threadIdx.x == 0) shared_mem[0] = max_val;
    __syncthreads();
    return shared_mem[0];
}

__global__ void attention_kernel(
    float *q, float *key_cache, float *value_cache, float *att, float *xb,
    int pos, int head_dim, int kv_dim, int max_seq_len, int kv_mul) {

    int head_id = blockIdx.x;
    int dim_id = threadIdx.x;
    int kv_head_id = head_id / kv_mul;

    int q_head_offset = head_id * head_dim;
    extern __shared__ unsigned char shm[];
    float *q_shm = reinterpret_cast<float*>(shm);
    float *att_shm = reinterpret_cast<float*>(shm + sizeof(float)*head_dim);
    q_shm[dim_id] = q[q_head_offset + dim_id];
    __syncthreads();

    for (int t = 0; t <= pos; ++t) {
        // {k|v}_val is assumed to point to correct layer
        float k_val = key_cache[t * kv_dim + kv_head_id * head_dim + dim_id];
        float attn_val = q_shm[dim_id] * k_val;

        float sum = block_reduce_sum(attn_val, att_shm);
        if (dim_id == 0) {
            att[head_id * max_seq_len + t] = sum / sqrt(static_cast<float>(head_dim));
        }
    }
    __syncthreads();

    //softmax
    float local_max = -FLT_MAX;
    for (int i = dim_id; i <= pos; i += blockDim.x) {
        local_max = fmaxf(local_max, att[head_id * max_seq_len + i]);
    }
    float global_max = block_reduce_max(local_max, att_shm);

    float local_denom = 0.0f;
    for (int i = dim_id; i <= pos; i += blockDim.x) {
        float exp_val = __expf(att[head_id * max_seq_len + i] - global_max);
        att[head_id * max_seq_len + i] = exp_val;
        local_denom += exp_val;
    }
    float global_denom = block_reduce_sum(local_denom, att_shm);

    for (int i = dim_id; i <= pos; i += blockDim.x) {
        att[head_id * max_seq_len + i] /= global_denom;
    }

    __syncthreads();

    float *xb_head = xb + head_id * head_dim;
    float out_val = 0.0f;
    for (int t = 0; t <= pos; ++t) {
        float v_val = value_cache[t * kv_dim + kv_head_id * head_dim + dim_id];
        out_val += att[head_id * max_seq_len + t] * v_val;
    }

    xb_head[dim_id] = out_val;
}

extern "C"
void cu::matmul_host_fp32(float *xout, float *x, const float *w, int d, int n) {
    // xout, x, w are already on device
    // w (n, d) @ x (d,) -> xout (n,)
    constexpr int WARPS_PER_BLOCK = 16;
    const int GRID = n;
    const int BLOCK = 32 * WARPS_PER_BLOCK;
    const int SHM_SIZE = sizeof(float) * WARPS_PER_BLOCK;

    matmul_kernel_fp32<WARPS_PER_BLOCK><<<GRID, BLOCK, SHM_SIZE>>>(xout, x, w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C"
void rmsnorm_host_fp32(float *xout, const float *x, const float *weight, int d, float eps) {
    const size_t BLOCK_SIZE = 256;
    const size_t GRID_SIZE = (d + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t SHM_SIZE = BLOCK_SIZE * sizeof(float);

    rmsnorm_kernel_fp32<<<GRID_SIZE, BLOCK_SIZE, SHM_SIZE>>>(xout, x, weight, d, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C"
void rope_host_fp32(float* d_q, float* d_k, int pos, float theta, int dim, int kv_dim, int head_dim)
{
    int total_threads = dim / 2;

    int threads_per_block = 256;

    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    rope_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_q, d_k, pos, theta, dim, kv_dim, head_dim
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C"
void silu_host(float* x, const float *x2, int d) {
    const size_t BLOCK_SIZE = 256;
    const size_t GRID_SIZE = (d + BLOCK_SIZE - 1) / BLOCK_SIZE;

    silu_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(x, x2, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

}

extern "C"
void residual_host(float* x, const float *res, int d) {
    const size_t BLOCK_SIZE = 256;
    const size_t GRID_SIZE = (d + BLOCK_SIZE - 1) / BLOCK_SIZE;

    residual_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(x, res, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C"
void attention_host(float *q, float *key_cache, float *value_cache, float *att, float *xb,
    int pos, int head_dim, int kv_dim, int max_seq_len, int kv_mul, int n_heads) {

    CUDA_CHECK(cudaMemset(att, 0, sizeof(float) * n_heads * max_seq_len));
    CUDA_CHECK(cudaMemset(xb, 0, sizeof(float) * n_heads * head_dim));

    const size_t BLOCK_SIZE = head_dim;
    const size_t GRID_SIZE = n_heads;
    const size_t SHM_SIZE = (BLOCK_SIZE + 32) * sizeof(float);

    attention_kernel<<<GRID_SIZE, BLOCK_SIZE, SHM_SIZE>>>(q, key_cache, value_cache, att, xb, pos, head_dim, kv_dim, max_seq_len, kv_mul);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
