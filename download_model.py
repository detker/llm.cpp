from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    local_dir="mistral",
    allow_patterns=["config.json", "tokenizer.json", "*.safetensors"],
)
