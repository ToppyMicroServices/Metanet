# Safe MetaNet

Safe MetaNet is a practical approach for updating model weights during inference when incoming data shifts away from training conditions. Instead of assuming every test-time update is helpful, it treats adaptation as a controlled process: make a candidate update, evaluate whether it helps, keep it only when accepted, and otherwise revert.

## Problem it solves: test-time degradation

Many models degrade at test time under distribution shift, and naive online adaptation can make this worse by reinforcing bad updates. Safe MetaNet is designed to reduce this risk by gating updates and preserving a known-good fallback state.

## How it differs from standard Test-Time Adaptation

Standard Test-Time Adaptation (TTA) often applies continual updates with limited safeguards, which can allow error accumulation. Safe MetaNet adds explicit safety controls around adaptation decisions so that updates are conditional, reversible, and bounded.

## Why rollback and scope expansion improve safety

- **Rollback (accept/reject):** each proposed update is checked; if it does not meet acceptance criteria, the model rolls back to the previous state instead of drifting further.
- **Adaptive scope expansion:** adaptation starts with a narrow, low-risk parameter scope and expands only when needed, limiting unintended changes while still allowing stronger correction under larger shifts.

Together, these mechanisms aim for stable improvement under shift while reducing the chance of catastrophic test-time performance drops.
