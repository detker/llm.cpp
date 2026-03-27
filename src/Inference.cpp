#include "Inference.hpp"

// w (d, n) @ x (n,) -> xout (d,)
void Inference::matmul(float *xout, float *x, const float *w, int n, int d) {
    for (int i = 0; i < d; ++i) {
        xout[i] = 0.0f;
        for (int j = 0; j < n; ++j) {
            xout[i] += w[i * n + j] * x[j];
        }
    }
}

// x_i: (x_i/sqrt(1/N * sum(x_j^2)+eps)) * w_i
void Inference::RMSnorm(float *xout, float *x, const float *w, int d, float eps) {
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

Inference::Inference(Config *config, RunState *runState, TransformerWeights *weights): config(config), runState(runState), weights(weights) {}

void Inference::RoPE(int pos) {
    const auto theta = config->rope_theta;
    const int kv_dim = (config->dim / config->n_heads) * config->n_kv_heads;

    for (int i = 0; i < config->dim; i += 2) {
        int head_i = i % config->head_dim;

        float freq = 1.0f / powf(theta, static_cast<float>(head_i) / static_cast<float>(config->head_dim));

        float val = freq * static_cast<float>(pos);
        float fcr = cosf(val);
        float fci = sinf(val);

        auto q0 = runState->q[i];
        auto q1 = runState->q[i + 1];
        runState->q[i] = q0 * fcr - q1 * fci;
        runState->q[i + 1] = q0 * fci + q1 * fcr;

        if (i < kv_dim) {
            auto k0 = runState->k[i];
            auto k1 = runState->k[i + 1];
            runState->k[i] = k0 * fcr - k1 * fci;
            runState->k[i + 1] = k0 * fci + k1 * fcr;
        }
    }
}

void Inference::UpdateKVCache(int layer_id, int pos) {
    int kv_dim = (config->dim / config->n_heads) * config->n_kv_heads;
    std::memcpy(runState->key_cache + layer_id * config->max_seq_len * kv_dim + pos * kv_dim, runState->k, sizeof(float) * kv_dim);
    std::memcpy(runState->value_cache + layer_id * config->max_seq_len * kv_dim + pos * kv_dim, runState->v, sizeof(float) * kv_dim);
}

void Inference::layer_forward(int layer_id, int pos) {
    auto &layer_weights = weights->layer_weights[layer_id];

    // 1. attn norm
    RMSnorm(runState->xb, runState->x, layer_weights.norm_att_weight, config->dim, config->norm_eps);

    int kv_dim = (config->dim / config->n_heads) * config->n_kv_heads;

    // 2. qkv projection
    matmul(runState->q, runState->xb, layer_weights.q_proj_weight, config->dim, config->dim);
    matmul(runState->k, runState->xb, layer_weights.k_proj_weight, config->dim, kv_dim);
    matmul(runState->v, runState->xb, layer_weights.v_proj_weight, config->dim, kv_dim);

    // 3. rotary embedding
    RoPE(pos);

    // 4. cache update
    UpdateKVCache(layer_id, pos);

    // 5. attention score calculation
    int kv_mul = config->n_heads / config->n_kv_heads; // how many q heads per one kv head
    std::memset(runState->att, 0, sizeof(float) * config->n_heads * config->max_seq_len);
    for (int h = 0; h < config->n_heads; ++h) {
        auto q_head_start = runState->q + h * config->head_dim;
        int kv_head_id = h / kv_mul; // which kv head should q head h use

        for (int t = 0; t <= pos; ++t) {
            auto k_start = runState->key_cache + layer_id * config->max_seq_len * kv_dim + t * kv_dim + kv_head_id * config->head_dim;

            for (int i = 0; i < config->head_dim; ++i) {
                runState->att[h * config->max_seq_len + t] += q_head_start[i] * k_start[i];
            }
            runState->att[h * config->max_seq_len + t] /= std::sqrt(static_cast<float>(config->head_dim));
        }

        softmax(runState->att + h * config->max_seq_len, pos + 1);

        float *xb_head = runState->xb + h * config->head_dim;
        std::memset(xb_head, 0, sizeof(float) * config->head_dim);

        for (int t = 0; t <= pos; ++t) {
            auto v_start = runState->value_cache + layer_id * config->max_seq_len * kv_dim + t * kv_dim + kv_head_id * config->head_dim;

            for (int i = 0; i < config->head_dim; ++i) {
                xb_head[i] += runState->att[h * config->max_seq_len + t] * v_start[i];
            }
        }
    }

    // 6. attention output projection
    // xb_head (d,) @ o_proj (d,d)
    matmul(runState->xb2, runState->xb, layer_weights.o_proj_weight, config->dim, config->dim);

    // residual
    for (int i=0; i < config->dim; ++i) {
        runState->x[i] += runState->xb2[i];
    }

    // 7. FFN
    RMSnorm(runState->xb, runState->x, layer_weights.mlp_norm_weight, config->dim, config->norm_eps);

    matmul(runState->hb, runState->xb, layer_weights.mlp_w1_weight, config->dim, config->hidden_dim);
    matmul(runState->hb2, runState->xb, layer_weights.mlp_w3_weight, config->dim, config->hidden_dim);
    silu(runState->hb, config->hidden_dim);
    for (int i=0; i < config->hidden_dim; ++i) {
        runState->hb[i] *= runState->hb2[i];
    }
    matmul(runState->xb2, runState->hb, layer_weights.mlp_w2_weight, config->hidden_dim, config->dim);

    for (int i=0; i < config->dim; ++i) {
        runState->x[i] += runState->xb2[i];
    }
}

void Inference::forward(int token, int pos) {
    // 1. input embedding
    auto embd_vec = weights->token_embd_table + token * config->dim;
    std::memcpy(runState->x, embd_vec, sizeof(float) * config->dim);

    // 2. layers loop
    for (int layer_id = 0; layer_id < config->n_layers; ++layer_id) {
        layer_forward(layer_id, pos);
    }

    // 3. final normalization
    RMSnorm(runState->xb, runState->x, weights->final_norm_weight, config->dim, config->norm_eps);

    // 4. final out projection
    matmul(runState->logits, runState->xb, weights->output_proj_weight, config->dim, config->vocab_size);
}
