#ifndef LLM_CPP_CU_HPP
#define LLM_CPP_CU_HPP

#include <cuda_runtime.h>

namespace cu
{
    extern "C"
    void matmul_host_fp32(float *xout, float *x, const float *w, int d, int n);

    extern "C"
    void rmsnorm_host_fp32(float *xout, const float *x, const float *weight, int d, float eps);

    extern "C"
    void rope_host_fp32(float* d_q, float* d_k, int pos, float theta, int dim, int kv_dim, int head_dim);

    extern "C"
    void silu_host(float* x, const float *x2, int d);

    extern "C"
    void residual_host(float* x, const float *res, int d);

    extern "C"
    void attention_host(float *q, float *key_cache, float *value_cache, float *att, float *xb,
        int pos, int head_dim, int kv_dim, int max_seq_len, int kv_mul, int n_heads);
}



#endif //LLM_CPP_CU_HPP
