#include "Sampler.hpp"

int Sampler::sample_argmax(float *logits, int vocab_size) {
    int max_index = 0;
    float max_value = logits[0];

    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_value) {
            max_value = logits[i];
            max_index = i;
        }
    }

    return max_index;
}
