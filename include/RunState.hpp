#ifndef LLM_CPP_RUNSTATE_HPP
#define LLM_CPP_RUNSTATE_HPP

#include "Config.hpp"
#include "TransformerWeights.hpp"


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

    WeightsVector<float> key_cache;   // [n_layers * max_seq_len * kv_dim]
    WeightsVector<float> value_cache; // [n_layers * max_seq_len * kv_dim]
};

#endif //LLM_CPP_RUNSTATE_HPP
