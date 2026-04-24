#ifndef LLM_CPP_UTILS_HPP
#define LLM_CPP_UTILS_HPP

#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <immintrin.h>
#include <nlohmann/json.hpp>

#include "Config.hpp"
#include "TransformerWeights.hpp"
#include "errorUtils.hpp"
#include "Tokenizer.hpp"

typedef uint16_t float16_t;

using json = nlohmann::json;


class MiscUtils {
public:
    struct ParseResult {
        std::string model_path;
        std::string txt;
        BackendType backend;
        float temperature = 0.0f;
        int max_seq_len = 0;
    };
    static ParseResult parseArgs(int argc, char **argv);
    static long long calcWeightSize(const char *model_path, int header_size, int tokenizer_size);
    struct Metrics {
        long long tokens_generated;
        double elapsed_seconds;
        long long weights_size_bytes;
    };
    static void printMetrics(Metrics &metrics);
};


class DataUtils {
public:
    DataUtils(const MiscUtils::ParseResult &args);
    ~DataUtils();
    Config getConfig();
    template <FP1632 T> std::unique_ptr<TransformerWeightsAuto<T>> mapModelWeights();
    std::unique_ptr<Tokenizer> getTokenizer();
    uint64_t getHeaderSize() const { return header_size; }
private:
    Config config;
    char *data;
    size_t file_size;
    int fd;
    json header;
    uint64_t header_size;
};

class MathUtils {
public:
    static void matmul(float *xout, float *x, const float *w, int n, int d);
    static void matmul_fp16(float *xout, float *x, const float16_t *w, int n, int d);
    static void RMSnorm(float *xout, float *x, const float *w, int d, float eps);
    static void softmax(float *x, int size);
    static void silu(float *x, int size);
};


#endif //LLM_CPP_UTILS_HPP