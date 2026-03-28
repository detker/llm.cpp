#include "Sampler.hpp"

#include <iostream>
#include <cstdlib>
#include <locale>

#include <Inference.hpp>

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

int Sampler::sample_temperature(float *logits, int vocab_size, float temperature) {
    // srand(time(NULL));
    // std::cout << (float)rand() / (float)RAND_MAX << " " << temperature << std::endl;
    if (temperature <= 0.0f) {
        return sample_argmax(logits, vocab_size);
    }

    // apply temperature scaling
    for (int i = 0; i < vocab_size; ++i) {
        logits[i] = logits[i] / temperature;
    }

    MathUtils::softmax(logits, vocab_size);

    float x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        sum += logits[i];
        if (x < sum) {
            return i;
        }
    }

    return vocab_size-1;
}
