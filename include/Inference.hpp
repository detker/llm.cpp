#ifndef LLM_CPP_INFERENCE_HPP
#define LLM_CPP_INFERENCE_HPP

#include <cmath>
#include <cstring>

#include "Config.hpp"
#include "RunState.hpp"
#include "TransformerWeights.hpp"

class Inference {
public:
    Inference(Config *config, RunState *runState, TransformerWeights *weights);
    ~Inference() = default;

    void matmul(float *xout, float *x, const float *w, int n, int d);
    void RMSnorm(float *xout, float *x, const float *w, int d, float eps);
    void softmax(float *x, int size);
    void silu(float *x, int size);

    void forward(int token, int pos);
private:
    Config *config;
    RunState *runState;
    TransformerWeights *weights;

    void layer_forward(int layer_id, int pos);
    void RoPE(int pos);
    void UpdateKVCache(int layer_id, int pos);
};

#endif //LLM_CPP_INFERENCE_HPP
