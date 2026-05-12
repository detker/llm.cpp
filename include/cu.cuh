#ifndef LLM_CPP_CU_HPP
#define LLM_CPP_CU_HPP

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "errorUtils.hpp"
#include "utils.hpp"

namespace cu
{
    extern "C"
    void matmul_host_fp32_hidim(float *xout, float *x, const float *w, int d, int n);

    extern "C"
    void matmul_host_fp32(float *xout, float *x, const float *w, int d, int n);

    extern "C"
    void fused_matmul_add_residual_host_fp32(float *xout, const float *x, const float *w, int d, int n);

    extern "C"
    void fused_ff1ff3matmul_silu_host_fp32(float *xout, const float *x, const float *w1, const float *w2, int d, int n);

    extern "C"
    void fused_qkv_matmuls_host_fp32(float *q, float *k, float *v, const float *x, const float *wq, const float *wk, const float *wv, int dim, int kv_dim);

    extern "C"
    void fused_rope_kvcache_update_host_fp32(float *q, float *k, float *v, int layer_id, int pos, float rope_theta, int dim, int kv_dim, int head_dim, float16_t *key_cache, float16_t *value_cache);

    extern "C"
    void rmsnorm_host_fp32(float *xout, const float *x, const float *weight, int d, float eps);

    extern "C"
    void rope_host_fp32(float* d_q, float* d_k, int pos, float theta, int dim, int kv_dim, int head_dim);

    extern "C"
    void silu_host(float* x, const float *x2, int d);

    extern "C"
    void residual_host(float* x, const float *res, int d);

    extern "C"
    void attention_host(float *q, float16_t *key_cache, float16_t *value_cache, float *att, float *xb,
        int pos, int head_dim, int kv_dim, int max_seq_len, int kv_mul, int n_heads, int n_kv_heads);
}



#endif //LLM_CPP_CU_HPP
