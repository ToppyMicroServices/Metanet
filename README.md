# Metanet
MetaNet: test-time weight editing with adaptive scope expansion and accept/reject rollback that improves models under distribution shift without labeled data

Includes a deterministic `AdaptiveScopeExpander` that:
- starts from the last adapter layer,
- tracks per-layer contribution scores,
- expands scope after configurable no-improvement patience,
- logs each scope change explicitly for ablations.
