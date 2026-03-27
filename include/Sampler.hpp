#ifndef LLM_CPP_SAMPLER_HPP
#define LLM_CPP_SAMPLER_HPP

class Sampler {
public:
    Sampler() = default;
    ~Sampler() = default;

    int sample_argmax(float *logits, int vocab_size);
};

#endif //LLM_CPP_SAMPLER_HPP
