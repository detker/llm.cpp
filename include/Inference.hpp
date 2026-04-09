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

#if defined(__AVX2__) && defined(__F16C__)
inline float half_to_float(float16_t x) {
    return _cvtsh_ss(x);
}
inline float16_t float_to_half(float x) {
    return _cvtss_sh(x, 0);
}
#else
inline float half_to_float(f16_t x) {
    ERR("float16 not supported on this platform");
    return 0;
}
inline f16_t float_to_half(float x) {
    ERR("float16 not supported on this platform");
    return 0;
}
#endif

template<FP1632 T>
class Inference {
public:
    Inference(Config *config, RunState *runState, std::unique_ptr<TransformerWeightsAuto<T>> weights);
    ~Inference() = default;

    void forward(int token, int pos);
    RunState *runState;
private:
    Config *config;

    std::unique_ptr<TransformerWeightsAuto<T>> weights;

    void layer_forward(int layer_id, int pos);
    void RoPE(int pos);
    void UpdateKVCache(int layer_id, int pos);

    void matmul(float *xout, float *x, const T *w, int n, int d) {
        if constexpr (std::same_as<T, float>) {
            MathUtils::matmul(xout, x, w, n, d);
        } else {
            MathUtils::matmul_fp16(xout, x, w, n, d);
        }
    }
};

#endif //LLM_CPP_INFERENCE_HPP
