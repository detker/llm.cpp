#include "Tokenizer.hpp"

#include <iostream>


Tokenizer::Tokenizer(char *tokens, int vocab_size, int bos_token, int eos_token): bos_token(bos_token), eos_token(eos_token) {
    this->root = std::make_unique<TrieNode>();
    this->tokens = std::vector<std::string>(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        this->tokens[i] = std::string(tokens);
        tokens += this->tokens[i].size() + 1; // move to the next token (including null terminator)

        auto p = this->root.get();
        for (unsigned char c : this->tokens[i]) {
            if (!p->children[c]) {
                p->children[c] = std::make_unique<TrieNode>();
            }
            p = p->children[c].get();
        }

        p->token_id = i;
    }
}

std::string Tokenizer::decode(int token) {
    if (token < 0 || token >= tokens.size()) {
        ERR("Invalid token ID");
    }
    return tokens[token];
}

std::vector<int> Tokenizer::encode(std::string txt, int max_tokens) {
    auto res = std::vector<int>();
    res.push_back(bos_token);
    std::string input = " " + txt;
    std::vector<std::string> words;
    for (size_t i = 0; i < std::min(static_cast<int>(input.size()), max_tokens); ++i) {
        words.emplace_back(1, input[i]);
    }

    bool any_merged = true;
    while (any_merged) {
        any_merged = false;

        for (size_t i = 0; i+1 < words.size(); ++i) {
            std::string merged = words[i] + words[i+1];

            auto p = this->root.get();
            for (unsigned char c : merged) {
                p = p->children[c].get();
                if (!p) {
                    break;
                }
            }
            if (p && p->token_id != -1) {
                words[i] = merged;
                words.erase(words.begin() + i + 1);
                any_merged = true;
                break; // restart from the beginning after a merge
            }
        }
    }

    for (const auto & word : words) {
        auto p = this->root.get();
        for (unsigned char c : word) {
            p = p->children[c].get();
            if (!p) {
                break;
            }
        }
        if (!p || p->token_id == -1) {
            ERR(("Token not found in vocabulary: " + word).c_str());
        }
        res.push_back(p->token_id);
    }

    return res;
}
