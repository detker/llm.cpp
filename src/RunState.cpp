#include "RunState.hpp"


RunState::RunState(Config &config) {
    x = new float[config.dim];
    xb = new float[config.dim];
    xb2 = new float[config.dim];
    hb = new float[config.hidden_dim];
    hb2 = new float[config.hidden_dim];
    q = new float[config.dim];
    k = new float[config.dim];
    v = new float[config.dim];

    att = new float[config.n_heads * config.max_seq_len];
    logits = new float[config.vocab_size];

    int kv_dim = (config.dim / config.n_heads) * config.n_kv_heads;
    key_cache = new float[config.n_layers * config.max_seq_len * kv_dim];
    value_cache = new float[config.n_layers * config.max_seq_len * kv_dim];

    this->config = &config;
}

RunState::~RunState() {
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
