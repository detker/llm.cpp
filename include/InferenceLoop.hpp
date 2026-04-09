#ifndef LLM_CPP_INFERENCELOOP_HPP
#define LLM_CPP_INFERENCELOOP_HPP
#include <atomic>
#include <csignal>
#include "Config.hpp"
#include "Inference.hpp"
#include "Timer.hpp"
#include "Tokenizer.hpp"
#include "Sampler.hpp"

class InferenceLoop {
public:
    static std::atomic<bool> interrupt_requested;

    InferenceLoop(Config *config, std::unique_ptr<Tokenizer> tokenizer, Sampler *sampler,
        std::variant<std::unique_ptr<Inference<float>>, std::unique_ptr<Inference<float16_t>>> inference_variant)
        : config(config), sampler(sampler), tokenizer(std::move(tokenizer)),
          inference_variant(std::move(inference_variant)) {
        std::signal(SIGINT, InterruptHandler);
        std::signal(SIGTERM, InterruptHandler);
    }

    static void InterruptHandler(int signum) {
        interrupt_requested.store(true);
    }

    std::pair<float, int> RunInferenceLoop(const std::string &prompt) {
        timerManager.SetTimer(config->backend == BackendType::GPU ? static_cast<Timer*>(&timerGpu) : static_cast<Timer*>(&timerCpu));

        std::vector<int> input_tokens_ids = tokenizer->encode(prompt, config->max_seq_len);
        for (uint i = 0; i < input_tokens_ids.size(); i++) {
            std::visit([&](auto& inf) { inf->forward(input_tokens_ids[i], i); }, inference_variant);
        }

        timerManager.Start();
        int tokens_generated = 0;
        for (uint i = input_tokens_ids.size(); i < config->max_seq_len; ++i) {
            if (interrupt_requested) {
                break;
            }

            int next_token = std::visit([this](auto& inf) {return sampler->sample_temperature(inf->runState->logits, config->vocab_size, config->temperature);}, inference_variant);
            if (next_token == config->eos_token_id) {
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

        return std::make_pair(timerManager.ElapsedSeconds(), tokens_generated);
    };

private:
    TimerCPU timerCpu;
    TimerGPU timerGpu;
    TimerManager timerManager;

    Config *config;
    Sampler *sampler;
    std::unique_ptr<Tokenizer> tokenizer;
    std::variant<std::unique_ptr<Inference<float>>, std::unique_ptr<Inference<float16_t>>> inference_variant;
};

#endif //LLM_CPP_INFERENCELOOP_HPP
