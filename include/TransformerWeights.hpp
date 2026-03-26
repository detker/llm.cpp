#ifndef LLM_CPP_TRANSFORMERWEIGHTS_HPP
#define LLM_CPP_TRANSFORMERWEIGHTS_HPP

struct TransformerWeights {
    const float *token_embd_table;

    const float *rms_att_weight;
    const float *q_proj_weight;
    const float *k_proj_weight;
    const float *v_proj_weight;
    const float *o_proj_weight;

    const float *ffn_up_proj_weight;
    const float *ffn_down_proj_weight;

    const float *rms_final_weight;
    const float *proj_out_weight;
};

#endif //LLM_CPP_TRANSFORMERWEIGHTS_HPP