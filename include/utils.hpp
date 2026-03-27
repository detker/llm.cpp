#ifndef LLM_CPP_UTILS_HPP
#define LLM_CPP_UTILS_HPP

#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <nlohmann/json.hpp>

#include "Config.hpp"
#include "TransformerWeights.hpp"
#include "errorUtils.hpp"
#include "Tokenizer.hpp"

using json = nlohmann::json;

class DataUtils {
public:
    DataUtils(const char *model_path);
    ~DataUtils();
    Config& getConfig();
    TransformerWeights& mapModelWeights();
    Tokenizer& getTokenizer();
private:
    Config config;
    TransformerWeights weights;
    Tokenizer tokenizer;
    char *data;
    size_t file_size;
    int fd;
    json header;
    uint64_t header_size;
};

#endif //LLM_CPP_UTILS_HPP