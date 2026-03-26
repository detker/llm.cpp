#include <iostream>
#include <vector>

#include "Config.hpp"

int main(int argc, char **argv) {
    Config config{
        .dim = 512,
        .n_layers = 12,
        .head_dim = 64,
        .hidden_dim = 2048,
        .n_heads = 8,
        .n_kv_heads = 8,
        .window_size = 512,
        .context_len = 512,
        .vocab_size = 30522
    };

    std::cout << config.context_len << std::endl;

    return 0;
}