from huggingface_hub import HfFolder, snapshot_download
from pathlib import Path
HfFolder.save_token('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')
mistral_models_path = "/home/whr-a/mistral_models/7B-v0.3"

# mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id="mistralai/Mistral-7B-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)