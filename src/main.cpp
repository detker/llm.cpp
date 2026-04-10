#include <iostream>
#include <vector>
#include <atomic>
#include <memory>
#include <variant>

#include "Config.hpp"
#include "Dispatcher.hpp"
#include "utils.hpp"
#include "TransformerWeights.hpp"
#include "RunState.hpp"
#include "Inference.hpp"
#include "Sampler.hpp"
#include "Tokenizer.hpp"
#include "Timer.hpp"
#include "InferenceLoop.hpp"

std::atomic<bool> InferenceLoop::interrupt_requested{false};

int main(int argc, char **argv) {
    std::mt19937 rng(std::random_device{}());

    MiscUtils::ParseResult args = MiscUtils::parseArgs(argc, argv);
    DataUtils dataUtils(args);
    Config config = dataUtils.getConfig();
    std::unique_ptr<Tokenizer> tokenizer = dataUtils.getTokenizer();
    CPURunState runState(&config);
    Sampler sampler(rng);

    auto [elapsed_seconds, tokens_generated] = RunInference<1>(std::move(args.txt), &config, &runState, &dataUtils, std::move(tokenizer), &sampler);

    if (tokens_generated == 0) {
        ERR("No tokens generated");
    }

    MiscUtils::Metrics metrics{
        .tokens_generated = tokens_generated,
        .elapsed_seconds = elapsed_seconds,
        .weights_size_bytes = MiscUtils::calcWeightSize(args.model_path.c_str(),
            dataUtils.getHeaderSize(), config.vocab_size * sizeof(char))
    };

    MiscUtils::printMetrics(metrics);

    return EXIT_SUCCESS;
}
