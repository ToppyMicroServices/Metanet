"""Configuration dataclass for Safe MetaNet.

All hyperparameters are grouped here.  A YAML file can be loaded with
:func:`load_config` and will be validated against the dataclass fields.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import List

import yaml


@dataclass
class SafeMetaNetConfig:
    # ------------------------------------------------------------------ model
    backbone: str = "mlp"          # "mlp" | "cnn"
    hidden_dim: int = 64           # hidden units for MLP layers
    num_layers: int = 3            # number of backbone layers
    input_dim: int = 32            # input feature dimension (MLP)
    output_dim: int = 10           # number of output classes

    # --------------------------------------------------------------- adapter
    adapter_type: str = "lora"     # "lora" | "linear_adapter"
    lora_rank: int = 4             # rank for LoRA decomposition
    adapter_bottleneck: int = 8    # bottleneck size for linear adapters

    # --------------------------------------------------------------- training
    lr: float = 1e-3               # learning rate for editable parameters
    num_steps: int = 1             # gradient steps per test sample
    trust_region_lambda: float = 0.01  # L2 penalty on adapter parameters

    # -------------------------------------------------------------- rollback
    rollback_threshold: float = 0.0   # accept only if metric improves by ≥ this

    # ---------------------------------------------------------- scope expansion
    # ordered list of layer names to include when expanding; the loop starts
    # with only the *first* entry and adds one entry per rejection.
    expansion_schedule: List[str] = field(
        default_factory=lambda: ["layer_2", "layer_1", "layer_0"]
    )

    # ------------------------------------------------------------ ablation
    disable_rollback: bool = False          # skip rollback (always accept)
    disable_scope_expansion: bool = False   # never expand editable scope

    # --------------------------------------------------------------- logging
    log_level: str = "INFO"   # Python logging level


def load_config(path: str) -> SafeMetaNetConfig:
    """Load a :class:`SafeMetaNetConfig` from a YAML file.

    Unknown keys in the YAML are silently ignored so that users can annotate
    config files with comments or extra metadata without causing errors.
    """
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}

    cfg = SafeMetaNetConfig()
    known = set(asdict(cfg).keys())
    for k, v in data.items():
        if k in known:
            setattr(cfg, k, v)
    return cfg


def save_config(cfg: SafeMetaNetConfig, path: str) -> None:
    """Serialise *cfg* to *path* as YAML."""
    with open(path, "w") as fh:
        yaml.dump(asdict(cfg), fh, default_flow_style=False, sort_keys=True)


def config_from_dict(d: dict) -> SafeMetaNetConfig:
    """Build a :class:`SafeMetaNetConfig` from a plain dictionary."""
    cfg = SafeMetaNetConfig()
    known = set(asdict(cfg).keys())
    for k, v in d.items():
        if k in known:
            setattr(cfg, k, v)
    return cfg
