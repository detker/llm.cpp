#ifndef LLM_CPP_CONFIG_HPP
#define LLM_CPP_CONFIG_HPP

struct Config {
    int dim;
    int n_layers;
    int head_dim;
    int hidden_dim;
    int n_heads;
    int n_kv_heads;
    int window_size;
    int context_len;
    int vocab_size;
};

#endif //LLM_CPP_CONFIG_HPP