#ifndef LLM_CPP_INFERENCE_HPP
#define LLM_CPP_INFERENCE_HPP

#include <cmath>
#include <cstring>
#include <memory>

#include "Config.hpp"
#include "RunState.hpp"
#include "TransformerWeights.hpp"
#include "utils.hpp"
#include "Sampler.hpp"
#include "cu.cuh"

#if defined(__AVX2__) && defined(__F16C__)
inline float half_to_float(float16_t x) {
    return _cvtsh_ss(x);
}
inline float16_t float_to_half(float x) {
    return _cvtss_sh(x, 0);
}
#else
inline float half_to_float(float16_t x) {
    ERR("float16 not supported on this platform");
    return 0;
}
inline float16_t float_to_half(float x) {
    ERR("float16 not supported on this platform");
    return 0;
}
#endif

template <FP1632 T>
class IInference {
public:
    IInference(Config *config, RunState *runState, std::unique_ptr<TransformerWeightsAuto<T>> weights)
    : runState(runState), config(config), weights(std::move(weights)) {}
    virtual ~IInference() = default;

    virtual float* forward(int token, int pos) = 0;
    RunState *runState;
protected:
    Config *config;

    std::unique_ptr<TransformerWeightsAuto<T>> weights;

    virtual void layer_forward(int layer_id, int pos) = 0;
    virtual void RoPE(int pos) = 0;
    virtual void UpdateKVCache(int layer_id, int pos) = 0;

    virtual void matmul(float *xout, float *x, const T *w, int n, int d) = 0;
};

template<FP1632 T>
class CPUInference : public IInference<T> {
public:
    CPUInference(Config *config, RunState *runState, std::unique_ptr<TransformerWeightsAuto<T>> weights);
    ~CPUInference() override = default;

    float* forward(int token, int pos) override;
private:
    using IInference<T>::config;
    using IInference<T>::runState;
    using IInference<T>::weights;

    void layer_forward(int layer_id, int pos) override;
    void RoPE(int pos) override;
    void UpdateKVCache(int layer_id, int pos) override;

    void matmul(float *xout, float *x, const T *w, int n, int d) override {
        if constexpr (std::same_as<T, float>) {
            MathUtils::matmul(xout, x, w, n, d);
        } else {
            MathUtils::matmul_fp16(xout, x, w, n, d);
        }
    }
};

template <FP1632 T>
class GPUInference : public IInference<T> {
public:
    GPUInference(Config *config, RunState *runState, std::unique_ptr<TransformerWeightsAuto<T>> weights);
    ~GPUInference() override = default;

    float* forward(int token, int pos) override;
private:
    using IInference<T>::config;
    using IInference<T>::runState;
    using IInference<T>::weights;

    void layer_forward(int layer_id, int pos) override;
    void UpdateKVCache(int layer_id, int pos) override;

    void RopeUpdateKVCache(int layer_id, int pos) {
        int kv_dim = (config->dim / config->n_heads) * config->n_kv_heads;
        if constexpr (std::same_as<T, float>) {
            cu::fused_rope_kvcache_update_host_fp32(runState->q, runState->k, runState->v, layer_id, pos, config->rope_theta, config->dim, kv_dim, config->head_dim,
                runState->key_cache + layer_id*config->max_seq_len*kv_dim + pos*kv_dim,
                runState->value_cache + layer_id*config->max_seq_len*kv_dim + pos*kv_dim);
        } else {
            ERR("FP16 not implemented yet.");
        }
    }

    void RoPE(int pos) override {
        int kv_dim = (config->dim / config->n_heads) * config->n_kv_heads;
        if constexpr (std::same_as<T, float>) {
            cu::rope_host_fp32(runState->q, runState->k, pos, config->rope_theta, config->dim, kv_dim, config->head_dim);
        } else {
            ERR("FP16 not implemented yet.");
        }
    }

    std::unique_ptr<float[]> logits_host;

    void matmul(float *xout, float *x, const T *w, int n, int d) override {
        if constexpr (std::same_as<T, float>) {
            auto v = std::min(n,d);
            if (d*n > v*v) cu::matmul_host_fp32_hidim(xout, x, w, n, d);
            else cu::matmul_host_fp32(xout, x, w, n, d);
        } else {
            ERR("FP16 matmul not implemented yet.");
        }
    }

    void fused_matmul_residuals(float *xout, const float *x, const T *w, int n, int d) {
        if constexpr (std::same_as<T, float>) {
            cu::fused_matmul_add_residual_host_fp32(xout, x, w, n, d);
        } else {
            ERR("FP16 fused matmul + residuals not implemented yet.");
        }
    }

    void fused_matmul_silu(float *xout, const float *x, const T *w1, const T *w2, int n, int d) {
        if constexpr (std::same_as<T, float>) {
            cu::fused_ff1ff3matmul_silu_host_fp32(xout, x, w1, w2, n, d);
        } else {
            ERR("FP16 fused matmul + silu not implemented yet.");
        }
    }

    void fused_qkv_matmuls(float *q_out, float *k_out, float *v_out, const float *x, const T *wq, const T *wk, const T *wv, int dim, int kvdim) {
        if constexpr (std::same_as<T, float>) {
            // cu::fused_qkv_matmul_host_fp32(q_out, k_out, v_out, x, wq, wk, wv, n, d);
            cu::fused_qkv_matmuls_host_fp32(q_out, k_out, v_out, x, wq, wk, wv, dim, kvdim);
        } else {
            ERR("FP16 fused qkv matmul not implemented yet.");
        }
    }

    void rmsnorm(float *xout, float *x, const float *w, int d, float eps) {
        if constexpr (std::same_as<T, float>) {
            cu::rmsnorm_host_fp32(xout, x, w, d, eps);
        } else {
            ERR("FP16 rmsnorm not implemented yet.");
        }
    }
};

#endif //LLM_CPP_INFERENCE_HPP
