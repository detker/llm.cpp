#ifndef LLM_CPP_CONFIG_HPP
#define LLM_CPP_CONFIG_HPP

#include <string>

enum ActType {
    GELU,
    SILU,
    RELU
};

enum BackendType {
    CPU,
    GPU
};

enum DType {
    FP32,
    FP16
};

struct Config {
    BackendType backend;
    ActType act_type;
    std::string arch;
    DType dtype;

    int dim;
    int n_layers;
    int head_dim;
    int hidden_dim;
    int n_heads;
    int n_kv_heads;
    int max_seq_len;
    int vocab_size;

    float norm_eps;
    float rope_theta;
    std::string norm_type;
    int rotary_dim;

    int bos_token_id;
    int eos_token_id;

    float temperature;
};

#endif //LLM_CPP_CONFIG_HPP