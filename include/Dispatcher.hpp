#ifndef LLM_CPP_DISPATCHER_HPP
#define LLM_CPP_DISPATCHER_HPP

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <utility>
#include "Config.hpp"
#include "Timer.hpp"
#include "errorUtils.hpp"
#include "Inference.hpp"
#include "InferenceLoop.hpp"
#include "TransformerWeights.hpp"
#include "RunState.hpp"


struct InferenceDispatcher {
    static InferenceLoop dispatch(Config *config, RunState *runState, DataUtils *dataUtils, std::unique_ptr<Tokenizer> tokenizer, Sampler *sampler) {
        std::variant<std::unique_ptr<IInference<float>>, std::unique_ptr<IInference<float16_t>>> inference_variant;

        if (config->backend == BackendType::CPU) {
            std::cout << "Running inference on CPU..." << std::endl;
            if (config->dtype == DType::FP16) {
                auto weights_ptr = dataUtils->mapModelWeights<float16_t>();
                inference_variant = std::make_unique<CPUInference<float16_t>>(config, runState, std::move(weights_ptr));
            } else {
                auto weights_ptr = dataUtils->mapModelWeights<float>();
                inference_variant = std::make_unique<CPUInference<float>>(config, runState, std::move(weights_ptr));
            }
        } else if (config->backend == BackendType::GPU) {
            ERR("GPU backend is not implemented yet");
        } else {
            ERR("Unsupported backend");
        }

        InferenceLoop inference_loop(config, std::move(tokenizer), sampler, std::move(inference_variant));
        return inference_loop;
    }
};

template<int CurrentD, int MaxD>
struct RuntimeDispatcher
{
    template<typename Func, typename... Args>
    static decltype(auto) dispatch(int D, Func&& func, Args&&... args)
    {
        if (D == CurrentD)
        {
            return std::forward<Func>(func).template operator()<CurrentD>(std::forward<Args>(args)...);
        }
        else
        {
            return RuntimeDispatcher<CurrentD + 1, MaxD>::dispatch(D, std::forward<Func>(func), std::forward<Args>(args)...);
        }
    }
};

template<int MaxD>
struct RuntimeDispatcher<MaxD, MaxD>
{
    template<typename Func, typename... Args>
    static decltype(auto) dispatch(int D, Func&& func, Args&&... args)
    {
        if (D == MaxD)
        {
            return std::forward<Func>(func).template operator()<MaxD>(std::forward<Args>(args)...);
        }
        // if D is out of bounds, handle error
        else
        {
            fprintf(stderr, "Error: Dimension %d out of bounds (1-%d)\n", D, MaxD);
            exit(EXIT_FAILURE);
        }
    }
};

struct InferenceLauncher
{
    std::string prompt;
    Config *config;
    RunState *runState;
    std::unique_ptr<Tokenizer> tokenizer;
    DataUtils *dataUtils;
    Sampler *sampler;

    InferenceLauncher(std::string prompt, Config *config, RunState *runState, DataUtils *dataUtils, std::unique_ptr<Tokenizer> tokenizer, Sampler *sampler)
        : prompt{std::move(prompt)}, config{config}, runState{runState}, dataUtils{dataUtils}, tokenizer{std::move(tokenizer)}, sampler{sampler} {}

    template<int RuntimeD>
    std::pair<float, int> operator()()
    {
        static_assert(RuntimeD >= 1, "Runtime dimension must be >= 1");
        auto inferenceLoop = InferenceDispatcher::dispatch(config, runState, dataUtils, std::move(tokenizer), sampler);
        return inferenceLoop.RunInferenceLoop(prompt);
    }
};

template<int D>
std::pair<float, int> RunInference(std::string prompt, Config *config, RunState *runState, DataUtils *dataUtils, std::unique_ptr<Tokenizer> tokenizer, Sampler *sampler)
{
    InferenceLauncher launcher{std::move(prompt), config, runState, dataUtils, std::move(tokenizer), sampler};

    return RuntimeDispatcher<1, 1>::dispatch(D, launcher);
}

#endif //LLM_CPP_DISPATCHER_HPP
