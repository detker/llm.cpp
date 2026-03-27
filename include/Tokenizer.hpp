#ifndef LLM_CPP_TOKENIZER_HPP
#define LLM_CPP_TOKENIZER_HPP

#include <string>
#include <unordered_map>
#include <vector>
#include <string>

#include "errorUtils.hpp"

class Tokenizer {
public:
    Tokenizer() = default;
    Tokenizer(char *token, int vocab_size, int bos_token, int eos_token);
    ~Tokenizer() = default;

    std::vector<int> encode(std::string txt, int max_tokens);
    std::string decode(int token);
private:
    std::vector<std::string> tokens;
    std::unordered_map<std::string, int> token2idx;
    int bos_token;
    int eos_token;

};

#endif //LLM_CPP_TOKENIZER_HPP
