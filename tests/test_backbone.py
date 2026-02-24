"""Tests for safe_metanet.backbone."""
import pytest
import torch

from safe_metanet.backbone import ToyMLP, ToyCNN, build_backbone, freeze_backbone
from safe_metanet.config import SafeMetaNetConfig


class TestToyMLP:
    def test_forward_shape(self):
        model = ToyMLP(input_dim=16, hidden_dim=32, output_dim=5, num_layers=3)
        x = torch.randn(4, 16)
        out = model(x)
        assert out.shape == (4, 5)

    def test_layer_names(self):
        model = ToyMLP(num_layers=3)
        assert model.layer_names() == ["layer_0", "layer_1", "layer_2"]

    def test_freeze_backbone(self):
        model = ToyMLP(input_dim=16, hidden_dim=32, output_dim=5, num_layers=2)
        freeze_backbone(model)
        for p in model.parameters():
            assert not p.requires_grad

    def test_min_layers(self):
        with pytest.raises(AssertionError):
            ToyMLP(num_layers=1)


class TestToyCNN:
    def test_forward_shape(self):
        model = ToyCNN(in_channels=1, hidden_dim=8, output_dim=5, spatial_size=8)
        x = torch.randn(2, 1, 8, 8)
        out = model(x)
        assert out.shape == (2, 5)

    def test_layer_names(self):
        model = ToyCNN()
        # conv1, conv2, fc
        assert len(model.layer_names()) == 3


class TestBuildBackbone:
    def test_build_mlp(self):
        cfg = SafeMetaNetConfig(backbone="mlp", input_dim=16, output_dim=4, num_layers=2)
        model = build_backbone(cfg)
        # After build, all params should be frozen
        for p in model.parameters():
            assert not p.requires_grad

    def test_build_cnn(self):
        cfg = SafeMetaNetConfig(backbone="cnn")
        model = build_backbone(cfg)
        for p in model.parameters():
            assert not p.requires_grad

    def test_unknown_backbone(self):
        cfg = SafeMetaNetConfig(backbone="transformer")
        with pytest.raises(ValueError):
            build_backbone(cfg)
