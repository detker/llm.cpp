#ifndef LLM_CPP_TRANSFORMERWEIGHTS_HPP
#define LLM_CPP_TRANSFORMERWEIGHTS_HPP

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

#endif //LLM_CPP_TRANSFORMERWEIGHTS_HPP
