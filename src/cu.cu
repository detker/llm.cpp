#include "cu.cuh"

#include <cfloat>
#include "errorUtils.hpp"


template<int WARPS_PER_BLOCK>
__global__ void matmul_kernel_fp32_hidim(float *xout, const float *x, const float *w, int n, int d) {
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

    if (lane_id == 0) warp_sums_shm[warp_id] = sum; // store warp sum in shared memory
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

template<int WARPS_PER_BLOCK>
__global__ void matmul_kernel_fp32(float *xout, const float *x, const float *w, int n, int d) {
    extern __shared__ unsigned char shm[];
    float *warp_sums_shm = (float*)shm;
    // [0, 1, ..., WARPS_PER_BLOCK-1]

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (row >= n) return;

    const float *w_row = w + row * d;
    float sum = 0.0f;

    int f4d = d / 4;
    const float4 *w_row_f4 = reinterpret_cast<const float4*>(w_row);
    const float4 *x_f4 = reinterpret_cast<const float4*>(x);
    for (int col = lane_id; col < f4d; col += 32) {
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

    if (lane_id == 0) warp_sums_shm[warp_id] = sum; // store warp sum in shared memory
    __syncthreads();
    if (warp_id == 0 && lane_id < WARPS_PER_BLOCK) {
        xout[blockIdx.x*WARPS_PER_BLOCK + lane_id] = warp_sums_shm[lane_id];
    }
}

template<int WARPS_PER_BLOCK>
__global__ void fused_matmul_kernel_add_residual_fp32(float *xout, const float *x, const float *w, int n, int d) {
    extern __shared__ unsigned char shm[];
    float *warp_sums_shm = (float*)shm;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (row >= n) return;

    const float *w_row = w + row * d;
    float sum = 0.0f;

    int f4d = d / 4;
    const float4 *w_row_f4 = reinterpret_cast<const float4*>(w_row);
    const float4 *x_f4 = reinterpret_cast<const float4*>(x);
    for (int col = lane_id; col < f4d; col += 32) {
        float4 w4 = w_row_f4[col];
        float4 x4 = x_f4[col];

        sum += (w4.x * x4.x +
                w4.y * x4.y +
                w4.z * x4.z +
                w4.w * x4.w);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) warp_sums_shm[warp_id] = sum;
    __syncthreads();
    int row_t = blockIdx.x * WARPS_PER_BLOCK + lane_id;
    if (warp_id == 0 && lane_id < WARPS_PER_BLOCK) {
        xout[row_t] += warp_sums_shm[lane_id]; // residual connection
    }
}

template<int WARPS_PER_BLOCK>
__global__ void fused_ff1ff3matmul_silu_kernel_fp32(float *xout, const float *x, const float *w1, const float *w2, int n, int d) {
    extern __shared__ unsigned char shm[];
    float *warp_sums_shm = (float*)shm;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (row >= n) return;

    const float *w1_row = w1 + row * d;
    const float *w2_row = w2 + row * d;
    float sum_1 = 0.0f, sum_2 = 0.0f;

    int f4d = d / 4;
    const float4 *w1_row_f4 = reinterpret_cast<const float4*>(w1_row);
    const float4 *w2_row_f4 = reinterpret_cast<const float4*>(w2_row);
    const float4 *x_f4 = reinterpret_cast<const float4*>(x);
    for (int col = lane_id; col < f4d; col += 32) {
        float4 w1_4 = w1_row_f4[col];
        float4 w2_4 = w2_row_f4[col];
        float4 x4 = x_f4[col];

        sum_1 += (w1_4.x * x4.x +
                  w1_4.y * x4.y +
                  w1_4.z * x4.z +
                  w1_4.w * x4.w);

        sum_2 += (w2_4.x * x4.x +
                  w2_4.y * x4.y +
                  w2_4.z * x4.z +
                  w2_4.w * x4.w);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_1 += __shfl_down_sync(0xffffffff, sum_1, offset);
        sum_2 += __shfl_down_sync(0xffffffff, sum_2, offset);
    }

    if (lane_id == 0) warp_sums_shm[warp_id] = (sum_1 / (1.0f + __expf(-sum_1))) * sum_2; // silu applied
    __syncthreads();

    if (warp_id == 0 && lane_id < WARPS_PER_BLOCK) {
        xout[blockIdx.x*WARPS_PER_BLOCK + lane_id] = warp_sums_shm[lane_id];
    }
}

template <int WARPS_PER_BLOCK>
__global__ void fused_qkv_matmuls_kernel_fp32(float *q, float *k, float *v,
                                              const float *x, const float *wq,
                                              const float *wk, const float *wv,
                                              int dim, int kv_dim) {
    extern __shared__ float shm_sums[];
    float *warp_sums_q_shm = shm_sums;
    float *warp_sums_k_shm = shm_sums + WARPS_PER_BLOCK;
    float *warp_sums_v_shm = shm_sums + WARPS_PER_BLOCK * 2;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    float sum_q = 0.0f, sum_k = 0.0f, sum_v = 0.0f;

    if (row < dim) {
        int f4d = dim / 4;
        const float4 *wq_row_f4 = reinterpret_cast<const float4*>(wq + row * dim);
        const float4 *wk_row_f4 = (row < kv_dim) ? reinterpret_cast<const float4*>(wk + row * dim) : nullptr;
        const float4 *wv_row_f4 = (row < kv_dim) ? reinterpret_cast<const float4*>(wv + row * dim) : nullptr;

        const float4 *x_f4 = reinterpret_cast<const float4*>(x);

        for (int col = lane_id; col < f4d; col += 32) {
            float4 x4 = x_f4[col];
            float4 wq_4 = wq_row_f4[col];

            sum_q += (wq_4.x * x4.x + wq_4.y * x4.y + wq_4.z * x4.z + wq_4.w * x4.w);

            if (row < kv_dim) {
                float4 wk_4 = wk_row_f4[col];
                float4 wv_4 = wv_row_f4[col];
                sum_k += (wk_4.x * x4.x + wk_4.y * x4.y + wk_4.z * x4.z + wk_4.w * x4.w);
                sum_v += (wv_4.x * x4.x + wv_4.y * x4.y + wv_4.z * x4.z + wv_4.w * x4.w);
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_q += __shfl_down_sync(0xffffffff, sum_q, offset);
            sum_k += __shfl_down_sync(0xffffffff, sum_k, offset);
            sum_v += __shfl_down_sync(0xffffffff, sum_v, offset);
        }

        if (lane_id == 0) {
            warp_sums_q_shm[warp_id] = sum_q;
            if (row < kv_dim) {
                warp_sums_k_shm[warp_id] = sum_k;
                warp_sums_v_shm[warp_id] = sum_v;
            }
        }
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < WARPS_PER_BLOCK) {
        int out_row = blockIdx.x * WARPS_PER_BLOCK + lane_id;
        if (out_row < dim) {
            q[out_row] = warp_sums_q_shm[lane_id];
            if (out_row < kv_dim) {
                k[out_row] = warp_sums_k_shm[lane_id];
                v[out_row] = warp_sums_v_shm[lane_id];
            }
        }
    }
}

template <int WARPS_PER_BLOCK>
__global__ void rmsnorm_kernel_fp32(float *xout, const float *x, const float *weight, int d, float eps) {
    extern __shared__ unsigned char shm[];
    float *warp_sums_shm = (float*)shm;

    int tid = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (tid >= d) return;

    float val = 0.0f;
    const float4 *x_f4 = reinterpret_cast<const float4*>(x);
    for (int i = tid; i < d/4; i += blockDim.x) {
        float4 val4 = x_f4[i];
        val += (val4.x * val4.x +
                val4.y * val4.y +
                val4.z * val4.z +
                val4.w * val4.w);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    // lane's 0 has correct sum for all warps

    if (lane_id == 0) warp_sums_shm[warp_id] = val;
    __syncthreads();

    val = (tid < WARPS_PER_BLOCK) ? warp_sums_shm[tid] : 0.0f;
    if (tid < 32) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }

    //tid = 0 now has correct sum
    if (tid == 0) {
        warp_sums_shm[0] = rsqrtf((val / d) + eps);
    }
    __syncthreads();

    float rms = warp_sums_shm[0];
    for (int i = tid; i < d; i += blockDim.x) {
        xout[i] = x[i] * rms * weight[i];
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

    // __syncthreads();
    //
    // float *xb_head = xb + head_id * head_dim;
    // float out_val = 0.0f;
    // for (int t = 0; t <= pos; ++t) {
    //     float v_val = value_cache[t * kv_dim + kv_head_id * head_dim + dim_id];
    //     out_val += att[head_id * max_seq_len + t] * v_val;
    // }
    //
    // xb_head[dim_id] = out_val;
}


__global__ void attention_vmix_kernel_fp32(float *xout, // [dim (n_heads * head_dim),]
                                           const float *att, // [n_heads, max_seq_len]
                                           const float *val_cache, // [max_seq_len, kv_dim (n_kv_heads * head_dim)] => [max_seq_len, n_kv_heads, head_dim]
                                           int head_dim,
                                           int n_heads,
                                           int n_kv_heads,
                                           int seq_len,
                                           int max_seq_len) {
    int head_id = blockIdx.x;
    int tid = threadIdx.x;
    // int seq_len_block_id = blockIdx.y;
    int group_size = n_heads / n_kv_heads;
    int g = head_id / group_size;
    int kv_stride = head_dim * n_kv_heads;

    const float *att_ptr = att + head_id * max_seq_len;
    const float *val_ptr = val_cache + g * head_dim;
    float *out_ptr = xout + head_id * head_dim;

    // int seq_len_block_size = seq_len / gridDim.y;
    // int start_t = seq_len_block_id * seq_len_block_size;

    __shared__ float shm[32];

    for (int i = tid; i < head_dim; i += 32) {
        if (threadIdx.y == 0) {
            shm[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        float val = 0.0f;
        // for (int t = start_t; t < start_t + seq_len_block_size && t < seq_len; ++t) {
        //     val += att_ptr[t] * val_ptr[kv_stride * t + i];
        // }
        for (int t = threadIdx.y; t < seq_len; t += blockDim.y) {
            val += att_ptr[t] * val_ptr[kv_stride * t + i];
        }
        atomicAdd(&shm[threadIdx.x], val);
        __syncthreads();
        if (threadIdx.y == 0) {
            out_ptr[i] = shm[threadIdx.x];
            shm[threadIdx.x] = 0.0f;
        }
    }
}


extern "C"
void cu::matmul_host_fp32_hidim(float *xout, float *x, const float *w, int d, int n) {
    // xout, x, w are already on device
    // w (n, d) @ x (d,) -> xout (n,)
    constexpr int WARPS_PER_BLOCK = 16;
    const int GRID = n;
    const int BLOCK = 32 * WARPS_PER_BLOCK;
    const int SHM_SIZE = sizeof(float) * WARPS_PER_BLOCK;

    matmul_kernel_fp32_hidim<WARPS_PER_BLOCK><<<GRID, BLOCK, SHM_SIZE>>>(xout, x, w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C"
void cu::matmul_host_fp32(float *xout, float *x, const float *w, int d, int n) {
    // xout, x, w are already on device
    // w (n, d) @ x (d,) -> xout (n,)
    constexpr int WARPS_PER_BLOCK = 16;
    const int GRID = (n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int BLOCK = 32 * WARPS_PER_BLOCK;
    const int SHM_SIZE = sizeof(float) * WARPS_PER_BLOCK;

    matmul_kernel_fp32<WARPS_PER_BLOCK><<<GRID, BLOCK, SHM_SIZE>>>(xout, x, w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C"
void cu::fused_matmul_add_residual_host_fp32(float *xout, const float *x, const float *w, int d, int n) {
    constexpr int WARPS_PER_BLOCK = 16;
    const int GRID = (n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int BLOCK = 32 * WARPS_PER_BLOCK;
    const int SHM_SIZE = sizeof(float) * WARPS_PER_BLOCK;

    fused_matmul_kernel_add_residual_fp32<WARPS_PER_BLOCK><<<GRID, BLOCK, SHM_SIZE>>>(xout, x, w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C"
void cu::fused_ff1ff3matmul_silu_host_fp32(float *xout, const float *x, const float *w1, const float *w2, int d, int n) {
    constexpr int WARPS_PER_BLOCK = 16;
    const int GRID = (n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int BLOCK = 32 * WARPS_PER_BLOCK;
    const int SHM_SIZE = sizeof(float) * WARPS_PER_BLOCK;

    fused_ff1ff3matmul_silu_kernel_fp32<WARPS_PER_BLOCK><<<GRID, BLOCK, SHM_SIZE>>>(xout, x, w1, w2, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C"
void fused_qkv_matmuls_host_fp32(float *q, float *k, float *v, const float *x, const float *wq, const float *wk, const float *wv, int dim, int kv_dim) {
    // d - config.dim
    // for Q (n=dim,d=dim)
    // for  K,V (n=kv_dim,d=dim)
    constexpr int WARPS_PER_BLOCK = 16;
    const int GRID = (dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int BLOCK = 32 * WARPS_PER_BLOCK;
    const int SHM_SIZE = sizeof(float) * WARPS_PER_BLOCK * 3;

    fused_qkv_matmuls_kernel_fp32<WARPS_PER_BLOCK><<<GRID, BLOCK, SHM_SIZE>>>(q, k, v, x, wq, wk, wv, dim, kv_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C"
void rmsnorm_host_fp32(float *xout, const float *x, const float *weight, int d, float eps) {
    const size_t WARPS_PER_BLOCK = 16;
    const size_t BLOCK_SIZE = WARPS_PER_BLOCK * 32;
    const size_t SHM_SIZE = sizeof(float) * WARPS_PER_BLOCK;

    rmsnorm_kernel_fp32<WARPS_PER_BLOCK><<<1, BLOCK_SIZE, SHM_SIZE>>>(xout, x, weight, d, eps);
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
    int pos, int head_dim, int kv_dim, int max_seq_len, int kv_mul, int n_heads, int n_kv_heads) {

    CUDA_CHECK(cudaMemset(att, 0, sizeof(float) * n_heads * max_seq_len));
    CUDA_CHECK(cudaMemset(xb, 0, sizeof(float) * n_heads * head_dim));

    const size_t BLOCK_SIZE = head_dim;
    const size_t GRID_SIZE = n_heads;
    const size_t SHM_SIZE = (BLOCK_SIZE + 32) * sizeof(float);

    attention_kernel<<<GRID_SIZE, BLOCK_SIZE, SHM_SIZE>>>(q, key_cache, value_cache, att, xb, pos, head_dim, kv_dim, max_seq_len, kv_mul);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // constexpr int max_t_per_thread = 256;
    constexpr int max_threads_per_block = 512;
    dim3 GRID_SIZE1;
    dim3 BLOCK_SIZE1;
    GRID_SIZE1.x = n_heads;
    // GRID_SIZE1.y = (pos+1 + max_t_per_thread - 1) / max_t_per_thread;
    BLOCK_SIZE1.x = 32; // warp size
    BLOCK_SIZE1.y = std::min(pos+1, max_threads_per_block / 32);
    attention_vmix_kernel_fp32<<<GRID_SIZE1, BLOCK_SIZE1>>>(xb, att, value_cache, head_dim, n_heads, n_kv_heads, pos+1, max_seq_len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
