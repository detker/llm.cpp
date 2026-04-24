import argparse
import os
import json
import safetensors
from safetensors.torch import save_file
import torch

SUPPORTED_DTYPES = ["fp32", "fp16"]

class Metadata:
  def __init__(self, config, dtype):
    if dtype not in SUPPORTED_DTYPES:
      raise Exception(f"Data type {dtype} is not supported, must be one of {SUPPORTED_DTYPES}")
    self.dtype = dtype
    self.dim = config["hidden_size"]
    self.hidden_dim = config["intermediate_size"]
    self.head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    self.n_layers = config["num_hidden_layers"]
    self.n_heads = config["num_attention_heads"]
    self.n_kv_heads = config.get("num_key_value_heads", config["num_attention_heads"])
    self.vocab_size = config["vocab_size"]
    self.max_seq_len = config["max_position_embeddings"]
    self.bos_token_id = config["bos_token_id"]
    self.eos_token_id = config["eos_token_id"]
    self.rope_theta = config.get("rope_theta", 10000.0)
    self.rotary_dim = int(self.head_dim * config.get("partial_rotary_factor", 1))
    self.norm_eps = config["rms_norm_eps"]
    self.act_type = config["hidden_act"]

  def to_dict(self):
    return {
      "arch": "MistralForCausalLM",
      "dtype": self.dtype,
      "dim": str(self.dim),
      "hidden_dim": str(self.hidden_dim),
      "head_dim": str(self.head_dim),
      "n_layers": str(self.n_layers),
      "n_heads": str(self.n_heads),
      "n_kv_heads": str(self.n_kv_heads),
      "vocab_size": str(self.vocab_size),
      "max_seq_len": str(self.max_seq_len),
      "bos_token_id": str(self.bos_token_id),
      "eos_token_id": str(self.eos_token_id),
      "rope_theta": str(self.rope_theta),
      "rotary_dim": str(self.rotary_dim),
      "norm_eps": str(self.norm_eps),
      "norm_type": "rmsnorm",
      "act_type": str(self.act_type),
    }

def load_tokens(tokenizer_path, vocab_size):
  tokens = [""] * vocab_size
  with open(tokenizer_path, "r") as f:
    tokenizer = json.load(f)

  vocab = tokenizer["model"]["vocab"]
  assert len(vocab) <= vocab_size

  for t, i in vocab.items():
    tokens[i] = t

  for added in tokenizer["added_tokens"]:
    tokens[added["id"]] = added["content"]

  for i, t in enumerate(tokens):
    t = t.replace('▁', ' ')
    b = t.encode('utf-8')
    b = b.replace(b"\0", b"\7")
    assert b.count(0) == 0
    tokens[i] = b

  return tokens

def load_weights(model_files, dtype_str, metadata, tie_word_embeddings):
  weights = {}
  for model_path in model_files:
    if model_path.endswith(".safetensors"):
      with safetensors.safe_open(model_path, framework="pt") as f:
        for k in f.keys():
          assert(k not in weights)
          weights[k] = f.get_tensor(k)

  # huggingface permutes WQ and WK, this function reverses it
  # see https://github.com/huggingface/transformers/blob/b132c1703eb1c8bd9dfa4ad6a9be2bfd6ef819e9/src/transformers/models/llama/convert_llama_weights_to_hf.py#L122
  def permute_reverse(w, heads, rotary_dim):
    head_dim = w.shape[0] // heads
    assert rotary_dim <= head_dim
    w = torch.unflatten(w, 0, (-1, head_dim))
    wr = w[:, :rotary_dim]
    wk = w[:, rotary_dim:]
    wr = torch.unflatten(wr, 1, (2, -1))
    wr = wr.transpose(1, 2)
    wr = wr.flatten(1, 2)
    w = torch.cat([wr, wk], dim=1)
    return torch.flatten(w, 0, 1)

  dtype = {"fp32": torch.float32, "fp16": torch.float16}[dtype_str]

  progress = 0
  def conv(t):
    nonlocal progress
    progress += 1
    print(f"\rConverting tensor {progress}: {t.shape}", end="", flush=True)
    return t.to(dtype)

  tensors = {}
  tensors["model.embed.weight"] = conv(weights["model.embed_tokens.weight"])

  for l in range(metadata.n_layers):
    tensors[f"model.layers.{l}.attn.norm.weight"] = weights[f"model.layers.{l}.input_layernorm.weight"].float()

    rotary_dim = metadata.rotary_dim
    n_heads = metadata.n_heads
    n_kv_heads = metadata.n_kv_heads

    tensors[f"model.layers.{l}.attn.wq.weight"] = conv(permute_reverse(weights[f"model.layers.{l}.self_attn.q_proj.weight"], n_heads, rotary_dim))
    tensors[f"model.layers.{l}.attn.wk.weight"] = conv(permute_reverse(weights[f"model.layers.{l}.self_attn.k_proj.weight"], n_kv_heads, rotary_dim))
    tensors[f"model.layers.{l}.attn.wv.weight"] = conv(weights[f"model.layers.{l}.self_attn.v_proj.weight"])
    tensors[f"model.layers.{l}.attn.wo.weight"] = conv(weights[f"model.layers.{l}.self_attn.o_proj.weight"])

    tensors[f"model.layers.{l}.mlp.norm.weight"] = weights[f"model.layers.{l}.post_attention_layernorm.weight"].float()
    tensors[f"model.layers.{l}.mlp.w1.weight"] = conv(weights[f"model.layers.{l}.mlp.gate_proj.weight"])
    tensors[f"model.layers.{l}.mlp.w2.weight"] = conv(weights[f"model.layers.{l}.mlp.down_proj.weight"])
    tensors[f"model.layers.{l}.mlp.w3.weight"] = conv(weights[f"model.layers.{l}.mlp.up_proj.weight"])

  tensors["model.norm.weight"] = weights["model.norm.weight"].float()
  if not tie_word_embeddings:
    tensors["model.output.weight"] = conv(weights["lm_head.weight"])

  print()
  return tensors

if __name__ == "__main__":
  argp = argparse.ArgumentParser()
  argp.add_argument("output", type=str)
  argp.add_argument("input", type=str)
  argp.add_argument("--dtype", type=str, default="fp16", choices=SUPPORTED_DTYPES)
  args = argp.parse_args()

  args.config = os.path.join(args.input, "config.json")
  if not os.path.exists(args.config):
    argp.error(f"config.json not found in {args.input}")

  args.tokenizer = os.path.join(args.input, "tokenizer.json")
  if not os.path.exists(args.tokenizer):
    argp.error(f"tokenizer.json not found in {args.input}")

  model_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".safetensors")]
  if len(model_files) == 0:
    argp.error(f"no .safetensors files found in {args.input}")

  with open(args.config, "r") as f:
    config = json.load(f)
    metadata = Metadata(config, args.dtype)

  tokens = load_tokens(args.tokenizer, metadata.vocab_size)
  tensors = load_weights(model_files, args.dtype, metadata, config.get("tie_word_embeddings", False))

  tensors["tokenizer.tokens"] = torch.cat([torch.tensor([x for x in b] + [0], dtype=torch.uint8) for b in tokens])

  print(f"Saving {len(tensors)} tensors...")
  save_file(tensors, args.output, metadata.to_dict())
