# Safe MetaNet

> **Test-time weight editing with adaptive scope expansion and rollback.**

Safe MetaNet is a minimal, research-oriented Python framework that demonstrates
a principled approach to **test-time adaptation (TTA)**:

* Adapt model parameters at test time using only self-supervised signals — **no labels, no offline fine-tuning**.
* Start with the smallest possible editable scope (e.g., the last-layer LoRA adapter).
* Accept an update only if it improves an online proxy metric.
* If not, **rollback** to the previous state and **expand** the editable scope to more layers.
* Repeat with strict trust-region constraints to prevent degradation.

---

## What is Safe MetaNet?

### Core idea

Standard inference uses a frozen model.  Standard TTA (e.g., Tent, TTT) fine-tunes
a subset of the model on every test batch, which risks catastrophic drift and requires
careful choice of which layers to adapt.

Safe MetaNet addresses both problems simultaneously:

| Property | Standard TTA | Safe MetaNet |
|---|---|---|
| Editable scope | Fixed (manually chosen) | **Adaptive** — starts minimal, expands on failure |
| Update acceptance | Always accepted | **Conditional** — only accepted if metric improves |
| Rollback | No | **Yes** — parameter snapshot restored on rejection |
| Trust region | Optional | **Built-in** L2 penalty on adapter parameters |
| Labels required | No | **No** |
| Offline training | Pretrain + possibly fine-tune | **Pretrain only** (adapters initialised to zero delta) |

### How it differs from standard TTA

| | Tent / TTT / SHOT | Safe MetaNet |
|---|---|---|
| **Scope** | BN layers, classifier | Last-layer LoRA → layer-by-layer expansion |
| **Accept/reject** | No gate — all updates applied | Explicit metric gate with rollback |
| **Scope expansion** | N/A | Layer-by-layer until improvement found |
| **Online proxy metric** | Entropy (minimised) | Entropy (pluggable) |
| **Trust region** | None | L2 penalty on adapter weights |

### Algorithm

```
for each test sample x:
    for each expansion level L in [last_layer, second_last, ...]:
        snap = snapshot(adapter_params[L])
        M_before = proxy_metric(model, x)
        adapter_params[L] -= lr * ∇ proxy_loss(model, x)   # gradient step
        M_after = proxy_metric(model, x)

        if M_after < M_before - threshold:
            ACCEPT   → keep update, log acceptance
            break
        else:
            REJECT   → restore(snap), log rejection
            expand scope to next layer
```

---

## Repository structure

```
safe_metanet/          Python package
├── __init__.py        Public API
├── config.py          SafeMetaNetConfig dataclass + YAML loader
├── backbone.py        Toy MLP and CNN backbones
├── adapters.py        LoRALinear, LinearAdapter, inject/snapshot/restore
├── loss.py            Entropy loss (self-supervised proxy), trust-region
├── loop.py            SafeMetaNet main loop
└── logger.py          Structured per-step logging

configs/
└── default.yaml       Default configuration

scripts/
└── run_demo.py        End-to-end demo (no dataset download)

tests/
├── test_backbone.py
├── test_adapters.py
└── test_loop.py
```

---

## Installation

```bash
pip install -e ".[dev]"
```

Only **PyTorch** and **PyYAML** are required at runtime.

---

## Quick start

```bash
# Run the demo with default settings
python scripts/run_demo.py

# Use linear adapters instead of LoRA
python scripts/run_demo.py --adapter linear_adapter

# Load a custom YAML config
python scripts/run_demo.py --config configs/default.yaml --steps 30

# Ablation: disable rollback (always accept updates)
python scripts/run_demo.py --disable-rollback

# Ablation: disable scope expansion (fixed minimal scope)
python scripts/run_demo.py --disable-scope-expansion
```

---

## Python API

```python
import torch
from safe_metanet import SafeMetaNetConfig, build_backbone, SafeMetaNet

# 1. Configure
cfg = SafeMetaNetConfig(
    backbone="mlp",
    num_layers=3,
    input_dim=32,
    output_dim=10,
    adapter_type="lora",      # "lora" | "linear_adapter"
    lora_rank=4,
    lr=1e-3,
    rollback_threshold=0.0,
    expansion_schedule=["layer_2", "layer_1", "layer_0"],
)

# 2. Build frozen backbone (adapters injected automatically)
model = build_backbone(cfg)
runner = SafeMetaNet(model, cfg)

# 3. Process test samples one by one (no labels)
for step in range(20):
    x = torch.randn(4, cfg.input_dim)          # synthetic input
    result = runner.step(x, sample_idx=step)
    print(result["action"], result["metric_before"], "→", result["metric_after"])

print(runner.summary())
```

---

## Configuration reference

All fields are defined in `safe_metanet/config.py` as a `dataclass`.
A YAML config can be loaded with `load_config(path)`.

| Field | Default | Description |
|---|---|---|
| `backbone` | `"mlp"` | `"mlp"` or `"cnn"` |
| `hidden_dim` | `64` | Hidden layer width |
| `num_layers` | `3` | Number of backbone layers |
| `input_dim` | `32` | Input feature dimension (MLP) |
| `output_dim` | `10` | Output classes |
| `adapter_type` | `"lora"` | `"lora"` or `"linear_adapter"` |
| `lora_rank` | `4` | LoRA decomposition rank |
| `adapter_bottleneck` | `8` | Bottleneck size for linear adapters |
| `lr` | `1e-3` | Adapter learning rate |
| `num_steps` | `1` | Gradient steps per test sample |
| `trust_region_lambda` | `0.01` | L2 penalty weight on adapter params |
| `rollback_threshold` | `0.0` | Minimum metric improvement to accept |
| `expansion_schedule` | `["layer_2","layer_1","layer_0"]` | Ordered scope expansion |
| `disable_rollback` | `False` | Ablation: always accept |
| `disable_scope_expansion` | `False` | Ablation: never expand scope |
| `log_level` | `"INFO"` | Python logging level |

---

## Adapters

### LoRA (`LoRALinear`)

Adds a low-rank side-path to a frozen linear layer:

```
y = W_frozen @ x  +  scale * B @ A @ x
```

`A` is Kaiming-initialised; `B` is zero-initialised so the delta is zero at the
start of adaptation.  Only `A` and `B` are trainable.

### Linear bottleneck adapter (`LinearAdapter`)

Adds a residual bottleneck path:

```
y = W_frozen @ x  +  W_up( ReLU( W_down @ x ) )
```

`W_up` is zero-initialised; delta is zero at start.

---

## Proxy metric

The default self-supervised proxy is **Shannon entropy** of the softmax
distribution:

```
H(x) = -∑_c  p_c  log p_c
```

A **lower entropy** means the model is making more confident predictions, which
is treated as a signal of better adaptation.  The loop accepts an update when
`H_after < H_before - threshold`.

A `"reconstruction"` loss is also provided as an alternative placeholder.

---

## Testing

```bash
pytest tests/ -v
```

All tests run on CPU with no dataset download required.

---

## Research notes

* **No labeled data** is used at any point during test-time adaptation.
* **No offline adaptation** step is needed; adapters start at a zero delta.
* **Rollback** prevents catastrophic forgetting: if an update hurts, it is
  discarded and the model is exactly restored.
* **Adaptive scope expansion** is conservative: it only adds more layers when
  the current scope is insufficient, keeping most of the model frozen.
* **Trust-region** (L2 penalty) prevents the adapter from moving too far from
  initialisation.

This prototype is intended as a starting point for a paper submission, not a
production system.  The proxy loss, backbone, and expansion schedule are all
swappable without changing the core loop.

---

## License

See [LICENSE](LICENSE).
