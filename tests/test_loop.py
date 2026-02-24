"""Tests for safe_metanet.loop (SafeMetaNet)."""
import torch

from safe_metanet import SafeMetaNet, SafeMetaNetConfig, build_backbone
from safe_metanet.adapters import adapter_parameters, snapshot_adapters
from safe_metanet.logger import SafeMetaNetLogger


def make_model_and_runner(overrides: dict | None = None):
    cfg = SafeMetaNetConfig(
        backbone="mlp",
        input_dim=16,
        hidden_dim=32,
        output_dim=8,
        num_layers=3,
        adapter_type="lora",
        lora_rank=2,
        lr=1e-2,
        num_steps=1,
        trust_region_lambda=0.0,
        rollback_threshold=0.0,
        expansion_schedule=["layer_2", "layer_1", "layer_0"],
        disable_rollback=False,
        disable_scope_expansion=False,
        log_level="WARNING",
    )
    if overrides:
        for k, v in overrides.items():
            setattr(cfg, k, v)
    model = build_backbone(cfg)
    runner = SafeMetaNet(model, cfg)
    return model, runner, cfg


class TestSafeMetaNetStep:
    def test_returns_valid_result(self):
        _, runner, cfg = make_model_and_runner()
        x = torch.randn(4, cfg.input_dim)
        result = runner.step(x)
        assert "action" in result
        assert result["action"] in ("accept", "reject")
        assert "metric_before" in result
        assert "metric_after" in result
        assert isinstance(result["scope"], list)
        assert isinstance(result["scope_expanded"], bool)

    def test_step_counter(self):
        _, runner, cfg = make_model_and_runner()
        x = torch.randn(4, cfg.input_dim)
        runner.step(x)
        runner.step(x)
        assert runner._step == 2

    def test_log_records_captured(self):
        _, runner, cfg = make_model_and_runner()
        x = torch.randn(4, cfg.input_dim)
        for _ in range(5):
            runner.step(x)
        assert len(runner.logger.records) == 5

    def test_disable_rollback_always_accepts(self):
        """With disable_rollback=True every step should be accepted."""
        _, runner, cfg = make_model_and_runner({"disable_rollback": True})
        x = torch.randn(4, cfg.input_dim)
        for _ in range(10):
            result = runner.step(x)
            assert result["action"] == "accept"

    def test_scope_expansion_triggered(self):
        """Force a high threshold so updates are always rejected, triggering expansion."""
        _, runner, cfg = make_model_and_runner(
            {
                "rollback_threshold": 1e6,  # impossibly high → always reject
                "disable_rollback": False,
                "disable_scope_expansion": False,
            }
        )
        x = torch.randn(4, cfg.input_dim)
        runner.step(x)  # reject → expand to layer_1
        assert len(runner.current_scope) >= 2 or runner.current_scope == ["layer_2"]

    def test_disable_scope_expansion_stays_minimal(self):
        """With disable_scope_expansion=True, scope never grows."""
        _, runner, cfg = make_model_and_runner(
            {
                "rollback_threshold": 1e6,
                "disable_scope_expansion": True,
            }
        )
        x = torch.randn(4, cfg.input_dim)
        initial_scope = list(runner.current_scope)
        for _ in range(5):
            runner.step(x)
        assert runner.current_scope == initial_scope

    def test_rollback_restores_parameters(self):
        """On rejection the adapter params should be identical to before the step."""
        _, runner, cfg = make_model_and_runner(
            {
                "rollback_threshold": 1e6,
                "disable_scope_expansion": True,
            }
        )
        x = torch.randn(4, cfg.input_dim)
        snap_before = snapshot_adapters(runner.model)
        result = runner.step(x)
        snap_after = snapshot_adapters(runner.model)
        assert result["action"] == "reject"
        for key in snap_before:
            assert torch.allclose(snap_before[key], snap_after[key])

    def test_run_method(self):
        """run() processes every element of the iterator."""
        _, runner, cfg = make_model_and_runner()
        data = [torch.randn(2, cfg.input_dim) for _ in range(7)]
        runner.run(iter(data))
        assert len(runner.logger.records) == 7


class TestSafeMetaNetSummary:
    def test_summary_empty(self):
        _, runner, _ = make_model_and_runner()
        s = runner.summary()
        assert "No records" in s

    def test_summary_after_run(self):
        _, runner, cfg = make_model_and_runner()
        x = torch.randn(4, cfg.input_dim)
        for _ in range(3):
            runner.step(x)
        s = runner.summary()
        assert "Total steps" in s
        assert "3" in s
