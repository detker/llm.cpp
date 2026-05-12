#ifndef LLM_CPP_RUNSTATE_HPP
#define LLM_CPP_RUNSTATE_HPP

#include "Config.hpp"
#include "TransformerWeights.hpp"

typedef uint16_t float16_t;

class RunState {
public:
    RunState(Config *config);

    WeightsVector<float> x;           // [dim]
    WeightsVector<float> xb;          // [dim]
    WeightsVector<float> xb2;         // [dim]
    WeightsVector<float> hb;          // [hidden_dim]
    WeightsVector<float> hb2;         // [hidden_dim]
    WeightsVector<float> q;           // [dim]
    WeightsVector<float> k;           // [dim]
    WeightsVector<float> v;           // [dim]
    WeightsVector<float> att;         // [n_heads * max_seq_len]
    WeightsVector<float> logits;      // [vocab_size]

    WeightsVector<float16_t> key_cache;   // [n_layers * max_seq_len * kv_dim] fp16 quantized
    WeightsVector<float16_t> value_cache; // [n_layers * max_seq_len * kv_dim] fp16 quantized
};

#endif //LLM_CPP_RUNSTATE_HPP
