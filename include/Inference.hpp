#ifndef LLM_CPP_INFERENCE_HPP
#define LLM_CPP_INFERENCE_HPP

#include <cmath>

class Inference {
public:
    void matmul(float *xout, float *x, float *w, int n, int d);
    void RMSnorm(float *xout, float *x, float *w, int d, float eps);
    void softmax(float *x, int size);
    void silu(float *x, int size);
};

#endif //LLM_CPP_INFERENCE_HPP
