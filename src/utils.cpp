#include "utils.hpp"


DataUtils::DataUtils(const MiscUtils::ParseResult &args): config(), header_size(0) {
    if ((fd = open(args.model_path.c_str(), O_RDONLY)) == -1) {
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

    config.temperature = args.temperature;
    config.backend = args.backend;
}

Config DataUtils::getConfig() {
    header_size = *reinterpret_cast<uint64_t *>(data);
    std::string json_str(data + sizeof(uint64_t), header_size);
    header = json::parse(json_str);

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

template<FP1632 T>
std::unique_ptr<TransformerWeightsAuto<T>> DataUtils::mapModelWeights() {
    auto weights = std::make_unique<TransformerWeightsAuto<T>>();
    char *weights_start = data + sizeof(uint64_t) + header_size;

    auto token_embd_table_offset = static_cast<size_t>(header["model.embed.weight"]["data_offsets"][0]);
    auto token_embd_table_ptr = reinterpret_cast<const T *>(weights_start + token_embd_table_offset);
    weights->token_embd_table = std::make_unique<WeightsVector<T>>(token_embd_table_ptr, config.vocab_size, config.dim, config.dtype, config.backend);

    auto final_norm_weight_offset = static_cast<size_t>(header["model.norm.weight"]["data_offsets"][0]);
    auto final_norm_weight_ptr = reinterpret_cast<const float *>(weights_start + final_norm_weight_offset);
    weights->final_norm_weight = std::make_unique<WeightsVector<float>>(final_norm_weight_ptr, 1, config.dim, DType::FP32, config.backend);

    auto output_proj_weight_offset = static_cast<size_t>(header["model.output.weight"]["data_offsets"][0]);
    auto output_proj_weight_ptr = reinterpret_cast<const T *>(weights_start + output_proj_weight_offset);
    weights->output_proj_weight = std::make_unique<WeightsVector<T>>(output_proj_weight_ptr, config.vocab_size, config.dim, config.dtype, config.backend);

    weights->layer_weights = std::make_unique<LayerWeightsAuto<T>[]>(config.n_layers);
    const int kv_dim = (config.dim / config.n_heads) * config.n_kv_heads;
    for (int i = 0; i < config.n_layers; ++i) {
        auto &layer = weights->layer_weights[i];
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        auto norm_att_weight_offset = static_cast<size_t>(header[prefix + "attn.norm.weight"]["data_offsets"][0]);
        auto norm_att_weight_ptr = reinterpret_cast<const float *>(weights_start + norm_att_weight_offset);
        layer.norm_att_weight = std::make_unique<WeightsVector<float>>(norm_att_weight_ptr, 1, config.dim, DType::FP32, config.backend);
        auto wk_att_weight_offset = static_cast<size_t>(header[prefix + "attn.wk.weight"]["data_offsets"][0]);
        auto wk_att_weight_ptr = reinterpret_cast<const T *>(weights_start + wk_att_weight_offset);
        layer.k_proj_weight = std::make_unique<WeightsVector<T>>(wk_att_weight_ptr, kv_dim, config.dim, config.dtype, config.backend);
        auto wo_att_weight_offset = static_cast<size_t>(header[prefix + "attn.wo.weight"]["data_offsets"][0]);
        auto wo_att_weight_ptr = reinterpret_cast<const T *>(weights_start + wo_att_weight_offset);
        layer.o_proj_weight = std::make_unique<WeightsVector<T>>(wo_att_weight_ptr, config.dim, config.dim, config.dtype, config.backend);
        auto wq_att_weight_offset = static_cast<size_t>(header[prefix + "attn.wq.weight"]["data_offsets"][0]);
        auto wq_att_weight_ptr = reinterpret_cast<const T *>(weights_start + wq_att_weight_offset);
        layer.q_proj_weight = std::make_unique<WeightsVector<T>>(wq_att_weight_ptr, config.dim, config.dim, config.dtype, config.backend);
        auto wv_att_weight_offset = static_cast<size_t>(header[prefix + "attn.wv.weight"]["data_offsets"][0]);
        auto wv_att_weight_ptr = reinterpret_cast<const T *>(weights_start + wv_att_weight_offset);
        layer.v_proj_weight = std::make_unique<WeightsVector<T>>(wv_att_weight_ptr, kv_dim, config.dim, config.dtype, config.backend);
        auto mlp_w1_weight_offset = static_cast<size_t>(header[prefix + "mlp.w1.weight"]["data_offsets"][0]);
        auto mlp_w1_weight_ptr = reinterpret_cast<const T *>(weights_start + mlp_w1_weight_offset);
        layer.mlp_w1_weight = std::make_unique<WeightsVector<T>>(mlp_w1_weight_ptr, config.hidden_dim, config.dim, config.dtype, config.backend);
        auto mlp_w2_weight_offset = static_cast<size_t>(header[prefix + "mlp.w2.weight"]["data_offsets"][0]);
        auto mlp_w2_weight_ptr = reinterpret_cast<const T *>(weights_start + mlp_w2_weight_offset);
        layer.mlp_w2_weight = std::make_unique<WeightsVector<T>>(mlp_w2_weight_ptr, config.dim, config.hidden_dim, config.dtype, config.backend);
        auto mlp_w3_weight_offset = static_cast<size_t>(header[prefix + "mlp.w3.weight"]["data_offsets"][0]);
        auto mlp_w3_weight_ptr = reinterpret_cast<const T *>(weights_start + mlp_w3_weight_offset);
        layer.mlp_w3_weight = std::make_unique<WeightsVector<T>>(mlp_w3_weight_ptr, config.hidden_dim, config.dim, config.dtype, config.backend);
        auto mlp_norm_weight_offset = static_cast<size_t>(header[prefix + "mlp.norm.weight"]["data_offsets"][0]);
        auto mlp_norm_weight_ptr = reinterpret_cast<const float *>(weights_start + mlp_norm_weight_offset);
        layer.mlp_norm_weight = std::make_unique<WeightsVector<float>>(mlp_norm_weight_ptr, 1, config.dim, DType::FP32, config.backend);
    }
    return weights;
}

std::unique_ptr<Tokenizer> DataUtils::getTokenizer() {
    char *weights_start = data + sizeof(uint64_t) + header_size;
    auto tokens = weights_start + static_cast<size_t>(header["tokenizer.tokens"]["data_offsets"][0]);
    std::unique_ptr<Tokenizer> tokenizer_ptr = std::make_unique<Tokenizer>(tokens, config.vocab_size, config.bos_token_id, config.eos_token_id);

    return tokenizer_ptr;
}

DataUtils::~DataUtils() {
    if (munmap(data, file_size)) {
        ERR("munmap error");
    }
    if (close(fd)) {
        ERR("close error");
    }
}


// w (d, n) @ x (n,) -> xout (d,)
void MathUtils::matmul(float *xout, float *x, const float *w, int n, int d) {
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < d; ++i) {
        xout[i] = 0.0f;
        for (int j = 0; j < n; ++j) {
            xout[i] += w[i * n + j] * x[j];
        }
    }
}

void MathUtils::matmul_fp16(float *xout, float *x, const float16_t *w, int n, int d) {
#if defined(__AVX2__) && defined(__F16C__)
    int i;
    assert(n % 16 == 0);
#pragma omp parallel for private(i)
    for (i = 0; i < d; ++i) {
        // intel primitives
        __m256 sumlo = _mm256_setzero_ps();
        __m256 sumhi = _mm256_setzero_ps();
        for (int j = 0; j < n; j += 16) {
            __m256i w_vec_16fp16 = _mm256_loadu_si256((__m256i*)&w[i * n + j]);
            __m128i w_vec_hi = _mm256_extractf128_si256(w_vec_16fp16, 1);
            __m128i w_vec_lo = _mm256_extractf128_si256(w_vec_16fp16, 0);
            __m256 x_vec_8b_lo = _mm256_loadu_ps(&x[j]);
            __m256 x_vec_8b_hi = _mm256_loadu_ps(&x[j+8]);
            __m256 w_vec_hi_256b = _mm256_cvtph_ps(w_vec_hi);
            __m256 w_vec_lo_256b = _mm256_cvtph_ps(w_vec_lo);
            sumlo = _mm256_fmadd_ps(w_vec_lo_256b, x_vec_8b_lo, sumlo);
            sumhi = _mm256_fmadd_ps(w_vec_hi_256b, x_vec_8b_hi, sumhi);
        }
        __m256 sum8 = _mm256_add_ps(sumlo, sumhi);
        __m128 sum4 = _mm_add_ps(_mm256_extractf128_ps(sum8, 1), _mm256_extractf128_ps(sum8, 0));
        __m128 sum1 = _mm_dp_ps(sum4, _mm_set1_ps(1.0f), 0xF1);
        xout[i] = _mm_cvtss_f32(sum1);
    }

#else
    ERR("FP16 matmul not supported on this platform");
#endif
}

// x_i: (x_i/sqrt(1/N * sum(x_j^2)+eps)) * w_i
void MathUtils::RMSnorm(float *xout, float *x, const float *w, int d, float eps) {
    float mean_square = 0.0f;
    for (int i = 0; i < d; ++i) {
        mean_square += x[i] * x[i];
    }
    mean_square /= d;

    float norm_factor = 1.0f / std::sqrt(mean_square + eps);
    int i;
    for (i = 0; i < d; ++i) {
        xout[i] = (x[i] * norm_factor) * w[i];
    }
}


void MathUtils::softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    float denom = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        denom += x[i];
    }

    for (int i = 0; i < size; ++i) {
        x[i] /= denom;
    }
}

void MathUtils::silu(float *x, int size) {
    for (int i = 0; i < size; ++i) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}


MiscUtils::ParseResult MiscUtils::parseArgs(int argc, char **argv) {
    if (argc < 5) {
        usage(argv[0]);
    }

    std::string model_path = argv[1];
    std::string txt = argv[2];
    float temp = atof(argv[3]);
    BackendType backend = std::string(argv[4]) == "cuda" ? BackendType::GPU : BackendType::CPU;
    int max_seq_len = argc > 5 ? atoi(argv[5]) : 0;

    return MiscUtils::ParseResult{.model_path = model_path, .txt = txt, .backend = backend, .temperature = temp, .max_seq_len = max_seq_len};
}

long long MiscUtils::calcWeightSize(const char *model_path, int header_size, int tokenizer_size) {
    struct stat model_stat;
    if (stat(model_path, &model_stat)) {
        ERR("stat error");
    }
    long long file_size = model_stat.st_size;
    long long weights_size_bytes = file_size - sizeof(uint64_t) - header_size - tokenizer_size;

    return weights_size_bytes;
}

void MiscUtils::printMetrics(MiscUtils::Metrics &metrics) {
    float tokens_per_sec = metrics.tokens_generated / metrics.elapsed_seconds;
    float latency_s_per_tok = metrics.elapsed_seconds / metrics.tokens_generated;

    double total_bytes_read = (double)metrics.weights_size_bytes * metrics.tokens_generated;
    double bandwidth_gb_per_sec = (total_bytes_read / (1024.0 * 1024.0 * 1024.0)) / metrics.elapsed_seconds;

    std::cout << std::endl;
    std::cout << "Generation time: " << metrics.elapsed_seconds << " seconds" << std::endl;
    std::cout << "Tokens generated: " << metrics.tokens_generated << std::endl;
    std::cout << "Tokens per second: " << tokens_per_sec << " tok/s" << std::endl;
    std::cout << "Latency: " << latency_s_per_tok << " s/tok" << std::endl;
    std::cout << "Bandwidth: " << bandwidth_gb_per_sec << " GB/s" << std::endl;
}

template std::unique_ptr<TransformerWeightsAuto<float>> DataUtils::mapModelWeights<float>();
template std::unique_ptr<TransformerWeightsAuto<float16_t>> DataUtils::mapModelWeights<float16_t>();
