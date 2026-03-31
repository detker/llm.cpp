#ifndef LLM_CPP_INFERENCE_HPP
#define LLM_CPP_INFERENCE_HPP

#include <cmath>
#include <cstring>

#include "Config.hpp"
#include "RunState.hpp"
#include "TransformerWeights.hpp"
#include "utils.hpp"

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

class Inference {
public:
    Inference(Config *config, RunState *runState, TransformerWeightsFP16 *weights);
    ~Inference() = default;
    //
    // void matmul(float *xout, float *x, const float *w, int n, int d);
    // void RMSnorm(float *xout, float *x, const float *w, int d, float eps);
    // void softmax(float *x, int size);
    // void silu(float *x, int size);

    void forward(int token, int pos);
private:
    Config *config;
    RunState *runState;
    TransformerWeightsFP16 *weights;

    void layer_forward(int layer_id, int pos);
    void RoPE(int pos);
    void UpdateKVCache(int layer_id, int pos);
};

#endif //LLM_CPP_INFERENCE_HPP
