#include "Inference.hpp"
#include <algorithm>


template<FP1632 T>
GPUInference<T>::GPUInference(Config *config, RunState *runState, std::unique_ptr<TransformerWeightsAuto<T>> weights)
    : IInference<T>(config, runState, std::move(weights)) { }

template<FP1632 T>
void GPUInference<T>::UpdateKVCache(int layer_id, int pos) {
    int kv_dim = (config->dim / config->n_heads) * config->n_kv_heads;
    CUDA_CHECK(cudaMemcpy(runState->key_cache + layer_id * config->max_seq_len * kv_dim + pos * kv_dim, runState->k, sizeof(float) * kv_dim, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(runState->value_cache + layer_id * config->max_seq_len * kv_dim + pos * kv_dim, runState->v, sizeof(float) * kv_dim, cudaMemcpyDeviceToDevice));
}

template <FP1632 T>
void GPUInference<T>::layer_forward(int layer_id, int pos) {
    auto &layer_weights = weights->layer_weights[layer_id];

    // 1. attn norm
    this->rmsnorm(runState->xb, runState->x, layer_weights.norm_att_weight->getData(), config->dim, config->norm_eps);

    // 2. qkv projection
    int kv_dim = (config->dim / config->n_heads) * config->n_kv_heads;
    // this->matmul(runState->q, runState->xb, layer_weights.q_proj_weight->getData(), config->dim, config->dim);
    // this->matmul(runState->k, runState->xb, layer_weights.k_proj_weight->getData(), config->dim, kv_dim);
    // this->matmul(runState->v, runState->xb, layer_weights.v_proj_weight->getData(), config->dim, kv_dim);
    this->fused_qkv_matmuls(runState->q, runState->k, runState->v, runState->xb, layer_weights.q_proj_weight->getData(), layer_weights.k_proj_weight->getData(), layer_weights.v_proj_weight->getData(), config->dim, kv_dim);

    // 3. rotary embedding
    this->RoPE(pos);

    // 4. cache update
    this->UpdateKVCache(layer_id, pos);

    // 5. attention score calculation
    int kv_mul = config->n_heads / config->n_kv_heads; // how many q heads per one kv head
    cu::attention_host(runState->q, runState->key_cache + layer_id * config->max_seq_len * kv_dim,
        runState->value_cache + layer_id * config->max_seq_len * kv_dim, runState->att, runState->xb,
        pos, config->head_dim, kv_dim, config->max_seq_len, kv_mul, config->n_heads, config->n_kv_heads);

    // 6. attention output projection + residual
    this->fused_matmul_residuals(runState->x, runState->xb, layer_weights.o_proj_weight->getData(), config->dim, config->dim);

    // 7. FFN
    this->rmsnorm(runState->xb, runState->x, layer_weights.mlp_norm_weight->getData(), config->dim, config->norm_eps);
    this->fused_matmul_silu(runState->hb, runState->xb, layer_weights.mlp_w1_weight->getData(), layer_weights.mlp_w3_weight->getData(), config->dim, config->hidden_dim);

    // 8. FFN output projection + residual
    this->fused_matmul_residuals(runState->x, runState->hb, layer_weights.mlp_w2_weight->getData(), config->hidden_dim, config->dim);
}

template<FP1632 T>
float* GPUInference<T>::forward(int token, int pos) {
    // 1. input embedding on device
    auto embd_vec = weights->token_embd_table->getData() + token * config->dim;

    if constexpr (std::same_as<T, float>) {
        CUDA_CHECK(cudaMemcpy(runState->x.getMutableData(), embd_vec, sizeof(float) * config->dim, cudaMemcpyDeviceToDevice));
    } else {
        ERR("FP16 not implemented yet");
    }

    // 2. layers loop
    for (int layer_id = 0; layer_id < config->n_layers; ++layer_id) {
        layer_forward(layer_id, pos);
    }

    // 3. final normalization
    this->rmsnorm(runState->xb, runState->x, weights->final_norm_weight->getData(), config->dim, config->norm_eps);

    // 4. final out projection
    this->matmul(this->runState->logits, this->runState->xb, this->weights->output_proj_weight->getData(), this->config->dim, this->config->vocab_size);

    // 5. copy logits from device to host
    if (!logits_host) {
        logits_host = std::make_unique<float[]>(config->vocab_size);
    }
    CUDA_CHECK(cudaMemcpy(logits_host.get(), this->runState->logits, sizeof(float) * config->vocab_size, cudaMemcpyDeviceToHost));

    return logits_host.get();
}

template class GPUInference<float>;
template class GPUInference<float16_t>;
