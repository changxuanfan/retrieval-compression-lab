---
name: QRAE implementation
about: Design train-only relevance supervision and a controlled RAE extension
title: "Implement QRAE training data and retrieval-aware objective"
labels: enhancement
---

## Scope

Implement the retrieval-supervised extension after agreeing on the RAE interface.
Keep preprocessing, frozen embeddings, index and evaluation shared.

## Design

- Positive/negative sampling and known-positive masks:
- Ranking formula, margin/temperature and loss weights:
- Reconstruction and regularization definitions:
- Training query access boundary and dev model selection:
- Related work / originality boundary:

## Acceptance

- [ ] Only train query IDs/qrels enter training and mining.
- [ ] Unknown documents are candidate negatives, not verified irrelevant labels.
- [ ] Ranking uses normalized latent; reconstruction uses unnormalized latent.
- [ ] Static shared encoder is explained accurately; documents remain indexable.
- [ ] Compare equal negative counts/steps or explicitly account for the difference.
- [ ] Include a rank-only control and plan a full-dimensional supervised control.
- [ ] Validate a small pilot before large sweeps.
- [ ] Tests catch query leakage and positive-as-negative mistakes.
- [ ] Record actual results and seeds; preserve final test for frozen evaluation.
