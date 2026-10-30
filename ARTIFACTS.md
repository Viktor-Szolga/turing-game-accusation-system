# Source-only repository and regeneration requirements

On 2026-10-30, datasets, serialized encodings/features, PyTorch checkpoints and saved models, experiment-result CSVs, generated training/run logs, Python bytecode caches, and notebook execution outputs were removed from **all Git history**. LFS tracking rules were also removed. Code, configuration inputs, documentation, notebook source/markdown, and dependency definitions were retained.

This is a storage cleanup, not a claim that every experiment can be reproduced without external inputs. The historical Git backup contains LFS pointers, **not the removed LFS payloads**. Old commit IDs change after the history rewrite; use a fresh clone rather than merging an old clone back into this repository.

## Inputs that are no longer bundled

- Raw message/game datasets, labels, user IDs, saved split IDs, and associated serialized metadata under `data/`.
- `data/message_encodings.pkl`, `data/BoW_encodings.pkl`, and other precomputed feature files.
- Model weights under checkpoint/trained-model folders, including `final_model/scaled_model_d.pth`.
- Training information, CSV results, and logs needed to reproduce individual published figures without rerunning the corresponding experiments.

The raw Turing Game data and original embedding service are not documented here as public downloadable assets. Obtain the appropriate original inputs separately before running the pipeline. No replacement data or model has been substituted.

## Recreate locally

1. Install dependencies as described in the README.
2. Supply the original raw data under the paths expected by the chosen experiment YAML and generation scripts. Preserve the original split/seed semantics if reproducing a published run.
3. Generate BoW features with `python data/generate_encodings/BoW_encoding.py` after supplying its input data.
4. For sentence embeddings, inspect `data/generate_encodings/encode_data.ipynb`. It requires the original local encoding server or a compatible replacement; it cannot be rerun directly in an isolated clone.
5. Run the relevant training configuration via `python train.py`, then run evaluation/plotting with the regenerated model and results available locally.

The analysis notebooks are retained without execution outputs. The evaluation notebook also depends on the original encoding service. Generated artifacts should remain untracked; `.gitignore` covers their usual filenames and directories. Git LFS is not required by this cleaned repository.
