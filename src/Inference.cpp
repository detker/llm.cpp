#include "Inference.hpp"

// w (d, n) @ x (n,) -> xout (d,)
void Inference::matmul(float *xout, float *x, float *w, int n, int d) {
    for (int i = 0; i < d; ++i) {
        xout[i] = 0.0f;
        for (int j = 0; j < n; ++j) {
            xout[i] += w[i * n + j] * x[j];
        }
    }
}

// x_i: (x_i/sqrt(1/N * sum(x_j^2)+eps)) * w_i
void Inference::RMSnorm(float *xout, float *x, float *w, int d, float eps) {
    float mean_square = 0.0f;
    for (int i = 0; i < d; ++i) {
        mean_square += x[i] * x[i];
    }
    mean_square /= d;

    float norm_factor = 1.0f / std::sqrt(mean_square + eps);
    for (int i = 0; i < d; ++i) {
        xout[i] = (x[i] * norm_factor) * w[i];
    }
}

void Inference::softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    float denom = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        denom += x[i];
    }

    for (int i = 0; i < size; ++i) {
        x[i] /= denom;
    }
}

void Inference::silu(float *x, int size) {
    for (int i = 0; i < size; ++i) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}
