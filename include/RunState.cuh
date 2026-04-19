#ifndef LLM_CPP_RUNSTATE_HPP
#define LLM_CPP_RUNSTATE_HPP

#include "Config.hpp"

class IRunState {
public:
    float *x; // [dim]
    float *xb; // [dim]
    float *xb2; // [dim]
    float *hb; // FFN [hidden_dim]
    float *hb2; // FFN [hidden_dim]
    float *q; // [dim]
    float *k; // [dim]
    float *v; // [dim]
    float *att; // [n_heads, seq_len]
    float *logits; //  [vocab_size]

    float *key_cache; // [n_layers, seq_len, kv_dim]
    float *value_cache; // [n_layers, seq_len, kv_dim]

    // IRunState() = delete;
    // ~IRunState() = delete;
};

class CPURunState : public IRunState {
public:
    CPURunState(Config *config);

    ~CPURunState();
};

class GPURunState : public IRunState {
public:
    GPURunState(Config *config);

    ~GPURunState();
};

#endif //LLM_CPP_RUNSTATE_HPP
