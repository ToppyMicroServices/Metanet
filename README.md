# Metanet
MetaNet: test-time weight editing with adaptive scope expansion and accept/reject rollback that improves models under distribution shift without labeled data

## Safe MetaNet core loop (PyTorch)

Core components are separated into:
- `/home/runner/work/Metanet/Metanet/metanet/model.py`: frozen base model + LoRA adapters
- `/home/runner/work/Metanet/Metanet/metanet/adaptation.py`: accept/reject + rollback + adaptive scope expansion loop
- `/home/runner/work/Metanet/Metanet/metanet/metrics.py`: metric utilities
- `/home/runner/work/Metanet/Metanet/metanet/utils.py`: snapshots, rollback helpers, and scope controls

Run focused tests with:

```bash
python -m unittest discover -v
```
