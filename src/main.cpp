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
    MiscUtils::ParseResult args = MiscUtils::parseArgs(argc, argv);
    DataUtils dataUtils(args.model_path.c_str());
    Config config = dataUtils.getConfig();
    TransformerWeights weights = dataUtils.mapModelWeights();
    Tokenizer tokenizer = dataUtils.getTokenizer();
    RunState runState = RunState(config);
    Inference inference(&config, &runState, &weights);
    Sampler sampler{};

    std::cout << "Mistral-7B-v0.2 | Input String: " << args.txt << " | Temperature: " << args.temperature << std::endl;
    printf("Model Answer: ");
    fflush(stdout);

    std::vector<int> input_tokens_ids = tokenizer.encode(args.txt, config.max_seq_len);
    for (uint i = 0; i < input_tokens_ids.size(); i++) {
        inference.forward(input_tokens_ids[i], i);
    }

    for (uint i = input_tokens_ids.size(); i < config.max_seq_len; ++i) {
        auto next_token = sampler.sample_temperature(runState.logits, config.vocab_size, args.temperature);
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
