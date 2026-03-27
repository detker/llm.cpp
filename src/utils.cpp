#include "utils.hpp"

DataUtils::DataUtils(const char *model_path) {
    if ((fd = open(model_path, O_RDONLY)) == -1) {
        ERR("open error");
    }

    struct stat sb{};
    if (fstat(fd, &sb) == -1) {
        ERR("fstat error");
    }

    file_size = sb.st_size;

    data = static_cast<char *>(mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (data == MAP_FAILED) {
        ERR("mmap error");
    }
}

Config& DataUtils::getConfig() {
    header_size = *reinterpret_cast<uint64_t *>(data);
    std::string json_str(data + sizeof(uint64_t), header_size);
    header = json::parse(json_str);

    for (auto &kv : header.items()) {
        std::cout << kv.key() << ": " << kv.value() << std::endl;
    }

    // Config config;
    auto metadata = header["__metadata__"];
    config.dtype = metadata["dtype"].get<std::string>() == "fp16" ? DType::FP16 : DType::FP32;
    config.dim = std::stoi(metadata["dim"].get<std::string>());
    config.n_layers = std::stoi(metadata["n_layers"].get<std::string>());
    config.head_dim = std::stoi(metadata["head_dim"].get<std::string>());
    config.hidden_dim = std::stoi(metadata["hidden_dim"].get<std::string>());
    config.n_heads = std::stoi(metadata["n_heads"].get<std::string>());
    config.n_kv_heads = std::stoi(metadata["n_kv_heads"].get<std::string>());
    config.max_seq_len = std::stoi(metadata["max_seq_len"].get<std::string>());
    config.vocab_size = std::stoi(metadata["vocab_size"].get<std::string>());
    config.norm_eps = std::stof(metadata["norm_eps"].get<std::string>());
    config.rope_theta = std::stof(metadata["rope_theta"].get<std::string>());
    config.bos_token_id = std::stoi(metadata["bos_token_id"].get<std::string>());
    config.eos_token_id = std::stoi(metadata["eos_token_id"].get<std::string>());

    return config;
}

TransformerWeights& DataUtils::mapModelWeights() {

    char *weights_start = data + sizeof(uint64_t) + header_size;

    auto token_embd_table_offset = static_cast<size_t>(header["model.embed.weight"]["data_offsets"][0]);
    weights.token_embd_table = reinterpret_cast<const float *>(weights_start + token_embd_table_offset);

    auto final_norm_weight_offset = static_cast<size_t>(header["model.norm.weight"]["data_offsets"][0]);
    weights.final_norm_weight = reinterpret_cast<const float *>(weights_start + final_norm_weight_offset);

    auto output_proj_weight_offset = static_cast<size_t>(header["model.output.weight"]["data_offsets"][0]);
    weights.output_proj_weight = reinterpret_cast<const float *>(weights_start + output_proj_weight_offset);

    weights.layer_weights = new LayerWeights[config.n_layers];
    for (int i = 0; i < config.n_layers; ++i) {
        auto &layer = weights.layer_weights[i];
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        auto norm_att_weight_offset = static_cast<size_t>(header[prefix + "attn.norm.weight"]["data_offsets"][0]);
        layer.norm_att_weight = reinterpret_cast<const float *>(weights_start + norm_att_weight_offset);
        auto wk_att_weight_offset = static_cast<size_t>(header[prefix + "attn.wk.weight"]["data_offsets"][0]);
        layer.k_proj_weight = reinterpret_cast<const float *>(weights_start + wk_att_weight_offset);
        auto wo_att_weight_offset = static_cast<size_t>(header[prefix + "attn.wo.weight"]["data_offsets"][0]);
        layer.o_proj_weight = reinterpret_cast<const float *>(weights_start + wo_att_weight_offset);
        auto wq_att_weight_offset = static_cast<size_t>(header[prefix + "attn.wq.weight"]["data_offsets"][0]);
        layer.q_proj_weight = reinterpret_cast<const float *>(weights_start + wq_att_weight_offset);
        auto wv_att_weight_offset = static_cast<size_t>(header[prefix + "attn.wv.weight"]["data_offsets"][0]);
        layer.v_proj_weight = reinterpret_cast<const float *>(weights_start + wv_att_weight_offset);
        auto mlp_w1_weight_offset = static_cast<size_t>(header[prefix + "mlp.w1.weight"]["data_offsets"][0]);
        layer.mlp_w1_weight = reinterpret_cast<const float *>(weights_start + mlp_w1_weight_offset);
        auto mlp_w2_weight_offset = static_cast<size_t>(header[prefix + "mlp.w2.weight"]["data_offsets"][0]);
        layer.mlp_w2_weight = reinterpret_cast<const float *>(weights_start + mlp_w2_weight_offset);
        auto mlp_w3_weight_offset = static_cast<size_t>(header[prefix + "mlp.w3.weight"]["data_offsets"][0]);
        layer.mlp_w3_weight = reinterpret_cast<const float *>(weights_start + mlp_w3_weight_offset);
        auto mlp_norm_weight_offset = static_cast<size_t>(header[prefix + "mlp.norm.weight"]["data_offsets"][0]);
        layer.mlp_norm_weight = reinterpret_cast<const float *>(weights_start + mlp_norm_weight_offset);
    }

    return weights;
}

Tokenizer& DataUtils::getTokenizer() {
    char *weights_start = data + sizeof(uint64_t) + header_size;
    auto tokens = weights_start + static_cast<size_t>(header["tokenizer.tokens"]["data_offsets"][0]);
    tokenizer = Tokenizer(tokens, config.vocab_size, config.bos_token_id, config.eos_token_id);
    return tokenizer;
}

DataUtils::~DataUtils() {
    delete[] weights.layer_weights;
    if (munmap(data, file_size)) {
        ERR("munmap error");
    }
    if (close(fd)) {
        ERR("close error");
    }
}
