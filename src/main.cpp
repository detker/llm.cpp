#include <iostream>
#include <vector>
#include <csignal>
#include <atomic>
#include <sys/stat.h>

#include "Config.hpp"
#include "utils.hpp"
#include "TransformerWeights.hpp"
#include "RunState.hpp"
#include "Inference.hpp"
#include "Sampler.hpp"
#include "Tokenizer.hpp"
#include "Timer.hpp"

std::atomic<bool> interrupt_requested(false);
TimerManager* g_timerManager = nullptr;

void signalHandler(int signum) {
    interrupt_requested = true;
    if (g_timerManager) {
        g_timerManager->Stop();
    }
    std::cout << "\n[Interrupted by user]" << std::endl;
}

int main(int argc, char **argv) {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    MiscUtils::ParseResult args = MiscUtils::parseArgs(argc, argv);
    DataUtils dataUtils(args.model_path.c_str());
    Config config = dataUtils.getConfig();
    TransformerWeightsFP16 weights = dataUtils.mapModelWeights();
    Tokenizer tokenizer = dataUtils.getTokenizer();
    RunState runState = RunState(config);
    Inference inference(&config, &runState, &weights);
    Sampler sampler{};

    TimerCPU timer;
    TimerManager timerManager;
    timerManager.SetTimer(&timer);
    g_timerManager = &timerManager;

    std::cout << "Mistral-7B-v0.2 | Input String: " << args.txt << " | Temperature: " << args.temperature << std::endl;
    printf("Model Answer: ");
    fflush(stdout);


    std::vector<int> input_tokens_ids = tokenizer.encode(args.txt, config.max_seq_len);
    for (uint i = 0; i < input_tokens_ids.size(); i++) {
        inference.forward(input_tokens_ids[i], i);

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

        std::string next_token_str = tokenizer.decode(next_token);

        printf("%s", next_token_str.c_str());
        fflush(stdout);

        inference.forward(next_token, i);
        tokens_generated++;
    }

    timerManager.Stop();
    float elapsed_seconds = timerManager.ElapsedSeconds();

    if (tokens_generated == 0) {
        std::cout << "\nNo tokens generated." << std::endl;
        return 0;
    }

    float tokens_per_sec = tokens_generated / elapsed_seconds;
    float latency_s_per_tok = elapsed_seconds / tokens_generated;

    // Get actual model file size using stat
    struct stat model_stat;
    if (stat(args.model_path.c_str(), &model_stat)) {
        ERR("stat error");
    }

    long long file_size = model_stat.st_size;

    long long weights_size_bytes = file_size - sizeof(uint64_t) - dataUtils.getHeaderSize() - dataUtils.getTokenizerSize();

    double total_bytes_read = (double)weights_size_bytes * tokens_generated;
    double bandwidth_gb_per_sec = (total_bytes_read / (1024.0 * 1024.0 * 1024.0)) / elapsed_seconds;

    std::cout << "Generation time: " << elapsed_seconds << " seconds" << std::endl;
    std::cout << "Tokens generated: " << tokens_generated << std::endl;
    std::cout << "Tokens per second: " << tokens_per_sec << " tok/s" << std::endl;
    std::cout << "Latency: " << latency_s_per_tok << " s/tok" << std::endl;
    std::cout << "Bandwidth: " << bandwidth_gb_per_sec << " GB/s" << std::endl;

    return 0;
}
