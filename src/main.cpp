#include <iostream>
#include <vector>
#include <csignal>
#include <atomic>
#include <memory>
#include <variant>

#include "Config.hpp"
#include "utils.hpp"
#include "TransformerWeights.hpp"
#include "RunState.hpp"
#include "Inference.hpp"
#include "Sampler.hpp"
#include "Tokenizer.hpp"
#include "Timer.hpp"

std::atomic<bool> interrupt_requested(false);
void signalHandler(int signum) {
    interrupt_requested = true;
}

int main(int argc, char **argv) {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    std::mt19937 rng(std::random_device{}());

    MiscUtils::ParseResult args = MiscUtils::parseArgs(argc, argv);
    DataUtils dataUtils(args.model_path.c_str());
    Config config = dataUtils.getConfig();
    std::unique_ptr<Tokenizer> tokenizer = dataUtils.getTokenizer();
    RunState runState(config);
    Sampler sampler(rng);

    std::variant<std::unique_ptr<Inference<float>>, std::unique_ptr<Inference<float16_t>>> inference_variant;

    if (config.dtype == DType::FP16) {
        auto weights_ptr = dataUtils.mapModelWeights<float16_t>();
        inference_variant = std::make_unique<Inference<float16_t>>(&config, &runState, std::move(weights_ptr));
    } else {
        auto weights_ptr = dataUtils.mapModelWeights<float>();
        inference_variant = std::make_unique<Inference<float>>(&config, &runState, std::move(weights_ptr));
    }

    TimerCPU timer;
    TimerManager timerManager;
    timerManager.SetTimer(&timer);

    std::cout << "Mistral-7B-v0.2 | Input String: " << args.txt << " | Temperature: " << args.temperature << std::endl;
    printf("Model Answer: ");
    fflush(stdout);

    std::vector<int> input_tokens_ids = tokenizer->encode(args.txt, config.max_seq_len);
    for (uint i = 0; i < input_tokens_ids.size(); i++) {
        std::visit([&](auto& inf) { inf->forward(input_tokens_ids[i], i); }, inference_variant);
    }

    timerManager.Start();
    int tokens_generated = 0;
    for (uint i = input_tokens_ids.size(); i < config.max_seq_len; ++i) {
        if (interrupt_requested) {
            break;
        }

        auto next_token = sampler.sample_temperature(runState.logits, config.vocab_size, args.temperature);
        if (next_token == config.eos_token_id) {
            std::cout << "End of sequence generated." << std::endl;
            break;
        }

        std::string next_token_str = tokenizer->decode(next_token);

        printf("%s", next_token_str.c_str());
        fflush(stdout);

        std::visit([&](auto& inf) { inf->forward(next_token, i); }, inference_variant);
        tokens_generated++;
    }
    timerManager.Stop();
    float elapsed_seconds = timerManager.ElapsedSeconds();

    if (tokens_generated == 0) {
        std::cout << "\nNo tokens generated." << std::endl;
        return 0;
    }

    MiscUtils::Metrics metrics{
        .tokens_generated = tokens_generated,
        .elapsed_seconds = elapsed_seconds,
        .weights_size_bytes = MiscUtils::calcWeightSize(args.model_path.c_str(),
            dataUtils.getHeaderSize(), config.vocab_size * sizeof(char))
    };

    MiscUtils::printMetrics(metrics);

    return 0;
}
