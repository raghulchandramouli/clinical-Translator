"""Modal-only model weight storage."""

from __future__ import annotations

from pathlib import Path

import modal

MODEL_DIR = Path("/models")
app = modal.App("clinical-translator-models")
volume = modal.Volume.from_name("clinical-translator-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub==1.25.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
    timeout=60 * 60,
    volumes={MODEL_DIR: volume},
)
def cache_model(repo_id: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id, revision=revision, local_dir=MODEL_DIR / repo_id)
    volume.commit()
    return path


@app.local_entrypoint()
def main(contract_path: str = "configs/contracts/curb65-medgemma-1.5-v1.json") -> None:
    from clinical_translator.contracts.validation import load

    contract = load(contract_path)
    for model in contract["models"].values():
        print(cache_model.remote(model["repo_id"], model["revision"]))
