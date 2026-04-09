#ifndef LLM_CPP_TRANSFORMERWEIGHTS_HPP
#define LLM_CPP_TRANSFORMERWEIGHTS_HPP

#include <concepts>
#include <memory>

typedef uint16_t float16_t;

template<typename T>
concept FP1632 = std::same_as<T, float> || std::same_as<T, float16_t>;

template<FP1632 T>
struct LayerWeightsAuto {
    const float *norm_att_weight;
    const T *q_proj_weight;
    const T *k_proj_weight;
    const T *v_proj_weight;
    const T *o_proj_weight;

    const float *mlp_norm_weight;
    const T *mlp_w1_weight;
    const T *mlp_w2_weight;
    const T *mlp_w3_weight;
};

template<FP1632 T>
struct TransformerWeightsAuto {
    const T *token_embd_table;

    std::unique_ptr<LayerWeightsAuto<T>[]> layer_weights;

    const float *final_norm_weight;
    const T *output_proj_weight;
};

#endif //LLM_CPP_TRANSFORMERWEIGHTS_HPP
