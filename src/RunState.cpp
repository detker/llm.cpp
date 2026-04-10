#include "RunState.hpp"


CPURunState::CPURunState(Config *config) : IRunState() {
    x = new float[config->dim];
    xb = new float[config->dim];
    xb2 = new float[config->dim];
    hb = new float[config->hidden_dim];
    hb2 = new float[config->hidden_dim];
    q = new float[config->dim];
    k = new float[config->dim];
    v = new float[config->dim];

    att = new float[config->n_heads * config->max_seq_len];
    logits = new float[config->vocab_size];

    int kv_dim = (config->dim / config->n_heads) * config->n_kv_heads;
    key_cache = new float[config->n_layers * config->max_seq_len * kv_dim];
    value_cache = new float[config->n_layers * config->max_seq_len * kv_dim];
}

CPURunState::~CPURunState() {
    delete[] x;
    delete[] xb;
    delete[] xb2;
    delete[] hb;
    delete[] hb2;
    delete[] q;
    delete[] k;
    delete[] v;
    delete[] att;
    delete[] logits;
    delete[] key_cache;
    delete[] value_cache;
}

GPURunState::GPURunState(Config *config): IRunState() {
    cudaMalloc(reinterpret_cast<void **>(&x), config->dim * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&xb), config->dim * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&xb2), config->dim * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&hb), config->hidden_dim * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&hb2), config->hidden_dim * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&q), config->dim * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&k), config->dim * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&v), config->dim * sizeof(float));

    cudaMalloc(reinterpret_cast<void **>(&att), config->n_heads * config->max_seq_len * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&logits), config->vocab_size * sizeof(float));

    int kv_dim = (config->dim / config->n_heads) * config->n_kv_heads;
    cudaMalloc(reinterpret_cast<void **>(&key_cache), config->n_layers * config->max_seq_len * kv_dim * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&value_cache), config->n_layers * config->max_seq_len * kv_dim * sizeof(float));
}

GPURunState::~GPURunState() {
    cudaFree(x);
    cudaFree(xb);
    cudaFree(xb2);
    cudaFree(hb);
    cudaFree(hb2);
    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(att);
    cudaFree(logits);
    cudaFree(key_cache);
    cudaFree(value_cache);
}
