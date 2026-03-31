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


class DataUtils {
public:
    DataUtils(const char *model_path);
    ~DataUtils();
    Config& getConfig();
    TransformerWeightsFP16& mapModelWeights();
    Tokenizer& getTokenizer();
    uint64_t getHeaderSize() const { return header_size; }
    uint64_t getTokenizerSize() const;
private:
    Config config;
    TransformerWeightsFP16 weights;
    Tokenizer tokenizer;
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

class MiscUtils {
public:
    struct ParseResult {
        std::string model_path;
        std::string txt;
        float temperature = 0.0f;
    };
    static ParseResult parseArgs(int argc, char **argv);
};

#endif //LLM_CPP_UTILS_HPP