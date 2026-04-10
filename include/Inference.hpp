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

template <FP1632 T>
class IInference {
public:
    IInference(Config *config, IRunState *runState, std::unique_ptr<TransformerWeightsAuto<T>> weights);
    virtual ~IInference() = default;

    virtual void forward(int token, int pos) = 0;
    IRunState *runState;
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
    CPUInference(Config *config, IRunState *runState, std::unique_ptr<TransformerWeightsAuto<T>> weights);
    ~CPUInference() override = default;

    void forward(int token, int pos) override;
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

#endif //LLM_CPP_INFERENCE_HPP
