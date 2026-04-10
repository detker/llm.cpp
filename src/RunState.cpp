#include "RunState.hpp"


RunState::RunState(Config *config)
    : x(1, config->dim, config->backend),
      xb(1, config->dim, config->backend),
      xb2(1, config->dim, config->backend),
      hb(1, config->hidden_dim, config->backend),
      hb2(1, config->hidden_dim, config->backend),
      q(1, config->dim, config->backend),
      k(1, config->dim, config->backend),
      v(1, config->dim, config->backend),
      att(config->n_heads, config->max_seq_len, config->backend),
      logits(1, config->vocab_size, config->backend),
      key_cache(config->n_layers * config->max_seq_len,
                (config->dim / config->n_heads) * config->n_kv_heads,
                config->backend),
      value_cache(config->n_layers * config->max_seq_len,
                  (config->dim / config->n_heads) * config->n_kv_heads,
                  config->backend)
{}
