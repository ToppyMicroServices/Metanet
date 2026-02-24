# Metanet
MetaNet: test-time weight editing with adaptive scope expansion and accept/reject rollback that improves models under distribution shift without labeled data

## Config-driven ablations

The following runtime options are intended to be controlled from config:

- `rollback.enabled` (on/off)
- `scope_expansion.enabled` (on/off)
- `trust_region.enabled` (on/off)
- `adaptation.steps` (integer step count)

A minimal baseline config is available at:

- `configs/base.yaml`

Minimal ablation configs (one toggle change per file):

- `configs/ablations/no_rollback.yaml`
- `configs/ablations/no_scope_expansion.yaml`
- `configs/ablations/no_trust_region.yaml`
- `configs/ablations/adaptation_steps_1.yaml`
