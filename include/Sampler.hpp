#ifndef LLM_CPP_SAMPLER_HPP
#define LLM_CPP_SAMPLER_HPP
#include <random>

class Sampler {
public:
    Sampler(std::mt19937 &rng);
    ~Sampler() = default;

    int sample_argmax(float *logits, int vocab_size);

    int sample_temperature(float *logits, int vocab_size, float temperature = 0.0f);
private:
    std::mt19937 rng;
};

#endif //LLM_CPP_SAMPLER_HPP
