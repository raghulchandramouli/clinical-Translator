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
```
