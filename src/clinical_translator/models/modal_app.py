"""Modal-only model caching and deterministic CURB-65 translation."""

from pathlib import Path
from typing import Any

import modal

MODEL_DIR = Path("/models")
REVISION_FILE = ".clinical-translator-revision"
app = modal.App("clinical-translator-models")
volume = modal.Volume.from_name("clinical-translator-models", create_if_missing=True)
cache_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub==1.25.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
runner_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub==1.25.1",
        "torch==2.13.0",
        "transformers==5.14.1",
    )
    .env(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
)


@app.function(
    image=cache_image,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def check_access(repo_id: str, revision: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(repo_id, revision=revision).sha


@app.function(
    image=cache_image,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
    timeout=60 * 60,
    volumes={MODEL_DIR: volume},
)
def cache_model(repo_id: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id, revision=revision, local_dir=MODEL_DIR / repo_id)
    (Path(path) / REVISION_FILE).write_text(revision, encoding="ascii")
    volume.commit()
    return path


@app.cls(
    image=runner_image,
    gpu="A10G",
    timeout=30 * 60,
    volumes={MODEL_DIR: volume},
)
class Translator:
    repo_id: str = modal.parameter()
    revision: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        path = MODEL_DIR / self.repo_id
        if (path / REVISION_FILE).read_text(encoding="ascii") != self.revision:
            raise RuntimeError(f"cached revision mismatch for {self.repo_id}")
        torch.manual_seed(20260728)
        torch.cuda.manual_seed_all(20260728)
        torch.use_deterministic_algorithms(True)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="eager",
        ).to("cuda").eval()

    @modal.method()
    def translate(
        self,
        prompts: list[str],
        generation: dict[str, Any],
    ) -> list[str]:
        import torch

        outputs = []
        for text in prompts:
            inputs = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": "{"},
                ],
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode():
                tokens = self.model.generate(
                    **inputs,
                    do_sample=generation["do_sample"],
                    max_new_tokens=generation["max_new_tokens"],
                )
            outputs.append(
                "{" + self.tokenizer.decode(
                    tokens[0, inputs["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                ).strip()
            )
        return outputs


def _smoke(contract: dict[str, Any], output: Path) -> None:
    from clinical_translator.data.generator import generate, write_jsonl
    from clinical_translator.models.protocol import parse, prompt, result

    wanted = {"00000", "10000", "01000", "00100", "00010", "00001"}
    records = [
        record
        for record in generate(contract)[0]
        if record["kind"] == "complete"
        and record["provenance"]["template_id"] == "t01"
        and record["provenance"]["combination"] in wanted
    ]
    records.sort(key=lambda record: record["id"])
    prompts = [prompt(record["prompt"]) for record in records]
    stored = []
    for role, model in contract["models"].items():
        runner = Translator(
            repo_id=model["repo_id"],
            revision=model["revision"],
        )
        raw_runs = [
            runner.translate.remote(
                prompts,
                contract["reproducibility"]["generation"],
            )
            for _ in range(2)
        ]
        runs = [[parse(raw) for raw in raw_run] for raw_run in raw_runs]
        if runs[0] != runs[1]:
            raise RuntimeError(f"{role} produced nondeterministic smoke outputs")
        for source, facts in zip(records, runs[0], strict=True):
            stored.append(
                result(
                    contract=contract,
                    source=source,
                    role=role,
                    facts=facts,
                )
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(stored, output)
    print(f"{len(stored)} validated outputs written to {output}")


@app.local_entrypoint()
def main(
    contract_path: str = "configs/contracts/curb65-medgemma-1.5-v1.json",
    download: bool = False,
    smoke: bool = False,
    output: str = "evidence/goal-04-smoke.jsonl",
) -> None:
    from clinical_translator.contracts.validation import load

    contract = load(contract_path)
    if smoke:
        _smoke(contract, Path(output))
        return
    function = cache_model if download else check_access
    for model in contract["models"].values():
        result = function.remote(model["repo_id"], model["revision"])
        if not download and result != model["revision"]:
            raise RuntimeError(f"revision mismatch for {model['repo_id']}")
        print(result)
