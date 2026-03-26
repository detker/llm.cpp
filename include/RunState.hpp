#ifndef LLM_CPP_RUNSTATE_HPP
#define LLM_CPP_RUNSTATE_HPP

struct RunState {
    // A. Tymczasowe bufory (tzw. Aktywacje) służące jako brudnopis dla aktualnie przetwarzanego tokena
    float* x;      // Główne wejście w warstwie (tzw. wektor resztkowy). Rozmiar: [dim]
    float* xb;     // Wejście po przejściu przez RMSNorm [dim]
    float* xb2;    // Kolejny bufor na wyniki pośrednie [dim]
    float* hb;     // Bufor dla ukrytego wymiaru w sieci FFN [hidden_dim]
    float* hb2;    // Drugi bufor dla sieci FFN [hidden_dim]
    float* q;      // Zapytanie (Query) po transformacji w bieżącej warstwie [dim]
    float* k;      // Klucz (Key) w bieżącej warstwie [dim]
    float* v;      // Wartość (Value) w bieżącej warstwie [dim]
    float* att;    // Wyniki z mechanizmu atencji (tzw. 'attention scores') [n_heads, seq_len]
    float* logits; // Końcowe prawdopodobieństwa dla słownika (wyjście przed samplerem) [vocab_size]

    // B. KV Cache - Bardzo ważne! Model pamięta tu historię konwersacji
    // Są to zarezerwowane duże bloki pamięci, w których zapamiętujemy klucze i wartości wszystkich
    // poprzednich tokenów, jakie przeszły przez model.
    float* key_cache;   // Rozmiar: [n_layers, seq_len, dim]
    float* value_cache; // Rozmiar: [n_layers, seq_len, dim]
};

#endif //LLM_CPP_RUNSTATE_HPP