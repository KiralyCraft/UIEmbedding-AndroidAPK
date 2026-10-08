# Agent Guide

This repository is intended to be extended by coding agents and humans without collapsing the experiment controls.

## Invariants

1. Global SSL positives must remain complete-screen views. Do not make arbitrary semantic crops equivalent to a whole Android screen.
2. Resolution dimensions used with DINOv3 must be divisible by the configured patch size.
3. Do not introduce all-other-record instance negatives on MoGUI without a false-negative study.
4. Preserve MobileViews repeated visits and their trace/sequence provenance.
5. Evaluation must remain package-restricted and must exclude the query itself.
6. New checkpoints must retain the resolved config and exact-resume state.
7. XML failures must disable XML loss for the record, not discard it from visual SSL by default.
8. Do not vendor or redistribute DINOv3 weights/source.

## Architecture boundaries

- `data/` may return tensors and metadata but must not instantiate models.
- `models/` must not download datasets or write experiment artifacts.
- `losses/` should be stateless callables where practical.
- `training/` owns DDP, AMP, checkpointing, logging, and iteration.
- `evaluation/` must be usable on frozen checkpoints without training dependencies such as W&B.
- `configs/base/` are fragments and are not required to validate as standalone experiments.

## Adding an experiment

1. Add a complete config below `configs/experiments/<name>/` using `extends`.
2. Give it a unique `runtime.output_dir` and W&B run name.
3. Add a granular script below `experiments/<name>/`.
4. Add a smoke-sized unit or integration test for any new data shape or loss.
5. Run:

```bash
PYTHONPATH=src pytest -q
./smoke_test.sh
find . -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

## Adding a supervised place phase

Implement it as a new model/loss pair rather than overloading XML targets. A batch should be organized around package and mined place groups, with multiple real views per place and hard negatives from the same package. Keep `structure_id`, `state_id`, Activity, trace distance, and action/transition context separate so label confidence can be represented explicitly.

The preferred path is:

```text
F5 teacher backbone/projector
  -> supervised multi-positive place head
  -> calibrated retrieval evaluation
  -> relational/dense distillation into F6 student
```

## Style

- Type annotations are required for public functions.
- Use dataclasses for configuration and structured outputs.
- Prefer explicit errors to silent fallback.
- Keep shell scripts `set -euo pipefail` compliant.
- Keep generated datasets, runs, weights, and caches out of source control.
