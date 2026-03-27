#include <iostream>
#include <vector>

#include "Config.hpp"
#include "utils.hpp"
#include "TransformerWeights.hpp"
#include "RunState.hpp"
#include "Inference.hpp"
#include "Sampler.hpp"
#include "Tokenizer.hpp"

int main(int argc, char **argv) {
    DataUtils dataUtils("../model.bin");
    Config config = dataUtils.getConfig();
    TransformerWeights weights = dataUtils.mapModelWeights();
    Tokenizer tokenizer = dataUtils.getTokenizer();
    RunState runState = RunState(config);
    Inference inference(&config, &runState, &weights);
    Sampler sampler{};


    std::string txt = "Hello world";
    std::vector<int> input_tokens_ids = tokenizer.encode(txt, config.max_seq_len);

    std::cout << "Input: ";
    for (uint i = 0; i < input_tokens_ids.size(); i++) {
        printf("%s ", tokenizer.decode(input_tokens_ids[i]).c_str());
        fflush(stdout);
        inference.forward(input_tokens_ids[i], i);
    }

    std::cout << std::endl << "Model: ";
    for (uint i = input_tokens_ids.size(); i < config.max_seq_len; ++i) {
        auto next_token = sampler.sample_argmax(runState.logits, config.vocab_size);
        if (next_token == config.eos_token_id) {
            std::cout << "End of sequence generated." << std::endl;
            break;
        }

        std::string next_token_str = tokenizer.decode(next_token);

        printf("%s", next_token_str.c_str());
        fflush(stdout);

        inference.forward(next_token, i);
    }
    std::cout << std::endl;

    return 0;
}
