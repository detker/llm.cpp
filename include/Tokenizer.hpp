#ifndef LLM_CPP_TOKENIZER_HPP
#define LLM_CPP_TOKENIZER_HPP

#include <string>
#include <unordered_map>
#include <vector>
#include <string>

#include "errorUtils.hpp"

struct TrieNode {
    int token_id; // if token_id is -1, it means it's not a complete token.

    TrieNode *children[256];

    TrieNode() {
        token_id = -1;
        for (auto & i : children) {
            i = nullptr;
        }
    }
};

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
    TrieNode *root;
    int bos_token;
    int eos_token;

};

#endif //LLM_CPP_TOKENIZER_HPP
