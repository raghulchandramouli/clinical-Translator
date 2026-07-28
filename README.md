# Clinical Translator

Auditable clinical-text translation into formally verified facts.

## Repository layout

```text
configs/                         Versioned experiment contracts
notebooks/                       Reproducible research notebooks
proofs/                          Lean specifications and proofs
src/clinical_translator/
├── contracts/                   Goal 01: scope and schemas
├── data/                        Goal 02: synthetic vignettes
├── models/                      Goal 04: model runners
├── evaluation/                  Goals 05 and 12: evaluation and counterexamples
├── interpretability/            Goals 06–09: features, circuits, interventions
└── assurance/                   Goals 03, 10, 11, and 13: oracle and certificates
tests/                           Runnable checks mirroring the source package
```

Generated datasets, activations, checkpoints, and reports belong in
`artifacts/`, which is not committed.

## Setup

```bash
uv sync
uv run python -m clinical_translator.contracts
```

The printed content-addressed contract reference identifies the frozen
experiment scope used by later artifacts.

Model weights stay on Modal:

```bash
uv run --with modal==1.5.3 modal setup
uv run --with modal==1.5.3 modal volume create clinical-translator-models
uv run --with modal==1.5.3 modal secret create huggingface HF_TOKEN="$HF_TOKEN"
uv run --with modal==1.5.3 modal run -m clinical_translator.models.modal_app
```

Accept the Google model terms on Hugging Face before caching the two pinned,
gated checkpoints. Never put the token or model weights in this repository.
