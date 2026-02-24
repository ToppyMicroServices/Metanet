"""Tests for safe_metanet.adapters."""
import torch
import torch.nn as nn

from safe_metanet.adapters import (
    LoRALinear,
    LinearAdapter,
    inject_adapters,
    adapter_parameters,
    snapshot_adapters,
    restore_adapters,
    wrap_linear,
)
from safe_metanet.backbone import ToyMLP, freeze_backbone
from safe_metanet.config import SafeMetaNetConfig


class TestLoRALinear:
    def test_forward_shape(self):
        base = nn.Linear(16, 8)
        lora = LoRALinear(base, rank=4)
        x = torch.randn(3, 16)
        out = lora(x)
        assert out.shape == (3, 8)

    def test_base_frozen(self):
        base = nn.Linear(16, 8)
        lora = LoRALinear(base, rank=4)
        for p in lora.base_layer.parameters():
            assert not p.requires_grad

    def test_lora_params_trainable(self):
        base = nn.Linear(16, 8)
        lora = LoRALinear(base, rank=4)
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_zero_init_delta(self):
        """At init lora_B == 0, so output equals base output."""
        base = nn.Linear(16, 8, bias=False)
        lora = LoRALinear(base, rank=4)
        x = torch.randn(2, 16)
        with torch.no_grad():
            base_out = base(x)
            lora_out = lora(x)
        assert torch.allclose(base_out, lora_out, atol=1e-6)


class TestLinearAdapter:
    def test_forward_shape(self):
        base = nn.Linear(16, 8)
        adapter = LinearAdapter(base, bottleneck=4)
        x = torch.randn(3, 16)
        out = adapter(x)
        assert out.shape == (3, 8)

    def test_base_frozen(self):
        base = nn.Linear(16, 8)
        adapter = LinearAdapter(base, bottleneck=4)
        for p in adapter.base_layer.parameters():
            assert not p.requires_grad

    def test_zero_init_delta(self):
        """At init W_up == 0, so output equals base output."""
        base = nn.Linear(16, 8, bias=False)
        adapter = LinearAdapter(base, bottleneck=4)
        x = torch.randn(2, 16)
        with torch.no_grad():
            base_out = base(x)
            adapter_out = adapter(x)
        assert torch.allclose(base_out, adapter_out, atol=1e-6)


class TestInjectAdapters:
    def test_inject_last_layer_lora(self):
        cfg = SafeMetaNetConfig(adapter_type="lora", lora_rank=2)
        model = ToyMLP(input_dim=16, hidden_dim=16, output_dim=4, num_layers=3)
        freeze_backbone(model)
        inject_adapters(model, cfg, layer_indices=[2])
        assert isinstance(model.layers[2], LoRALinear)
        assert isinstance(model.layers[0], nn.Linear)

    def test_inject_linear_adapter(self):
        cfg = SafeMetaNetConfig(adapter_type="linear_adapter", adapter_bottleneck=4)
        model = ToyMLP(input_dim=16, hidden_dim=16, output_dim=4, num_layers=3)
        freeze_backbone(model)
        inject_adapters(model, cfg, layer_indices=[2])
        assert isinstance(model.layers[2], LinearAdapter)

    def test_default_scope_is_last_layer(self):
        cfg = SafeMetaNetConfig(adapter_type="lora")
        model = ToyMLP(input_dim=16, hidden_dim=16, output_dim=4, num_layers=3)
        freeze_backbone(model)
        inject_adapters(model, cfg)   # no layer_indices → last layer
        assert isinstance(model.layers[2], LoRALinear)
        assert isinstance(model.layers[0], nn.Linear)


class TestSnapshotRestore:
    def test_roundtrip(self):
        cfg = SafeMetaNetConfig(adapter_type="lora", lora_rank=2)
        model = ToyMLP(input_dim=16, hidden_dim=16, output_dim=4, num_layers=2)
        freeze_backbone(model)
        inject_adapters(model, cfg)

        snap = snapshot_adapters(model)
        # Corrupt parameters
        for p in adapter_parameters(model):
            p.data.fill_(999.0)

        restore_adapters(model, snap)
        for key, val in snap.items():
            # Find the tensor again after restoration
            pass  # restore succeeded without error

        # Verify values are restored
        snap2 = snapshot_adapters(model)
        for key in snap:
            assert torch.allclose(snap[key], snap2[key])
