"""Safe MetaNet – public API.

Typical usage::

    from safe_metanet import SafeMetaNetConfig, build_backbone, SafeMetaNet

    cfg = SafeMetaNetConfig(backbone="mlp", num_layers=3)
    model = build_backbone(cfg)
    runner = SafeMetaNet(model, cfg)

    import torch
    for _ in range(10):
        x = torch.randn(4, cfg.input_dim)
        result = runner.step(x)
        print(result["action"], result["metric_before"], "→", result["metric_after"])

    print(runner.summary())
"""

from .config import SafeMetaNetConfig, load_config, save_config, config_from_dict
from .backbone import ToyMLP, ToyCNN, build_backbone, freeze_backbone
from .adapters import (
    LoRALinear,
    LinearAdapter,
    inject_adapters,
    adapter_parameters,
    snapshot_adapters,
    restore_adapters,
)
from .loss import entropy_loss, compute_proxy_metric, differentiable_proxy_loss
from .logger import SafeMetaNetLogger, UpdateRecord
from .loop import SafeMetaNet

__all__ = [
    # config
    "SafeMetaNetConfig",
    "load_config",
    "save_config",
    "config_from_dict",
    # backbone
    "ToyMLP",
    "ToyCNN",
    "build_backbone",
    "freeze_backbone",
    # adapters
    "LoRALinear",
    "LinearAdapter",
    "inject_adapters",
    "adapter_parameters",
    "snapshot_adapters",
    "restore_adapters",
    # loss
    "entropy_loss",
    "compute_proxy_metric",
    "differentiable_proxy_loss",
    # logger
    "SafeMetaNetLogger",
    "UpdateRecord",
    # loop
    "SafeMetaNet",
]
