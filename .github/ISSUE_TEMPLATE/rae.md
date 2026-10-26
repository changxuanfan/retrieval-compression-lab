---
name: RAE implementation
about: Independently reproduce RAE and connect it to the shared evaluation pipeline
title: "Implement RAE with an explicit reproduction protocol"
labels: enhancement
---

## Scope

Implement RAE independently using standard PyTorch layers/optimizers. Reuse the
frozen embedding cache, full-corpus search and metrics already in the repository.

## Protocol decisions

- Paper version / reference commit:
- Encoder/decoder, bias, centering and normalization:
- Explicit regularization target and loss reduction:
- Optimizer, update count and reproducibility settings:
- Differences from the current centered linear AE:

## Acceptance

- [ ] Distinguish faithful reproduction from matched-architecture control.
- [ ] Do not copy the reference project's core training pipeline.
- [ ] Use no dev/test relevance labels for fitting.
- [ ] Preserve Torch/FAISS process isolation.
- [ ] Meaningful objective/shape/save-load tests and a small pilot pass.
- [ ] Record geometry and retrieval results separately, with complete configuration.
- [ ] Update README with actual scope and limitations; no placeholder scores.
