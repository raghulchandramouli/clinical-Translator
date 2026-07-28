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

Generate the finite synthetic dataset:

```bash
uv run python -m clinical_translator.data
```

Check the executable and formally verified scorers:

```bash
uv run python -m unittest tests/test_oracle.py -v
$HOME/.elan/bin/lean -o /tmp/CURB65.olean proofs/ClinicalTranslator/CURB65.lean
LEAN_PATH=/tmp $HOME/.elan/bin/leanchecker CURB65
```

Model weights stay on Modal:

```bash
uv run --with modal==1.5.3 modal setup
uv run --with modal==1.5.3 modal volume create clinical-translator-models
uv run --with modal==1.5.3 modal secret create huggingface HF_TOKEN="$HF_TOKEN"
uv run --with modal==1.5.3 modal run -m clinical_translator.models.modal_app
```

Accept the Google model terms on Hugging Face before caching the two pinned,
gated checkpoints. The command above checks access without downloading weights;
add `--download` to cache them in the Modal Volume. Never put the token or model
weights in this repository.

Run the repeated MedGemma/control smoke check:

```bash
uv run --with modal==1.5.3 modal run --env dev -m clinical_translator.models.modal_app --smoke
```

It writes only validated five-Boolean outputs and reproducibility metadata to
`evidence/goal-04-smoke.jsonl`; generated explanations are rejected and never
stored.
