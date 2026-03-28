#ifndef LLM_CPP_SAMPLER_HPP
#define LLM_CPP_SAMPLER_HPP

class Sampler {
public:
    Sampler() = default;
    ~Sampler() = default;

    int sample_argmax(float *logits, int vocab_size);

    int sample_temperature(float *logits, int vocab_size, float temperature = 0.0f);
};

#endif //LLM_CPP_SAMPLER_HPP
