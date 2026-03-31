#ifndef LLM_CPP_TRANSFORMERWEIGHTS_HPP
#define LLM_CPP_TRANSFORMERWEIGHTS_HPP

#include <cstdint>

typedef uint16_t float16_t;

struct LayerWeights {
    const float *norm_att_weight;
    const float *q_proj_weight;
    const float *k_proj_weight;
    const float *v_proj_weight;
    const float *o_proj_weight;

    const float *mlp_norm_weight;
    const float *mlp_w1_weight;
    const float *mlp_w2_weight;
    const float *mlp_w3_weight;
};

struct TransformerWeights {
    const float *token_embd_table;

    LayerWeights *layer_weights;

    const float *final_norm_weight;
    const float *output_proj_weight;
};

struct LayerWeightsFP16 {
    const float *norm_att_weight;
    const float16_t *q_proj_weight;
    const float16_t *k_proj_weight;
    const float16_t *v_proj_weight;
    const float16_t *o_proj_weight;

    const float *mlp_norm_weight;
    const float16_t *mlp_w1_weight;
    const float16_t *mlp_w2_weight;
    const float16_t *mlp_w3_weight;
};

struct TransformerWeightsFP16 {
    const float16_t *token_embd_table;

    LayerWeightsFP16 *layer_weights;

    const float *final_norm_weight;
    const float16_t *output_proj_weight;
};

#endif //LLM_CPP_TRANSFORMERWEIGHTS_HPP
