import unittest

import torch
from torch import nn

from metanet.adaptation import SafeMetaNetLoop
from metanet.model import SafeMetaNetModel


def _build_model() -> SafeMetaNetModel:
    torch.manual_seed(0)
    layer1 = nn.Linear(4, 4, bias=False)
    layer2 = nn.Linear(4, 2, bias=False)
    return SafeMetaNetModel([layer1, layer2], rank=2)


def _batch():
    torch.manual_seed(1)
    return torch.randn(5, 4), torch.randn(5, 2)


class SafeMetaNetLoopTests(unittest.TestCase):
    def test_base_model_is_frozen(self):
        model = _build_model()
        self.assertTrue(all(not parameter.requires_grad for parameter in model.base_layers.parameters()))

    def test_reject_rolls_back_and_expands_scope(self):
        model = _build_model()
        batch = _batch()

        def metric_fn(model, batch):
            # Metric is best when adapter weights stay near zero.
            with torch.no_grad():
                penalty = 0.0
                for adapter in model.adapter_modules():
                    for parameter in adapter.parameters():
                        penalty += float(torch.sum(parameter * parameter))
                return -penalty

        loop = SafeMetaNetLoop(model=model, metric_fn=metric_fn, learning_rate=0.5)
        first_adapter_before = [parameter.detach().clone() for parameter in model.adapters[0].parameters()]

        result = loop.step(batch)

        self.assertFalse(result.accepted)
        self.assertEqual(result.active_scope, 2)
        for before, after in zip(first_adapter_before, model.adapters[0].parameters()):
            self.assertTrue(torch.allclose(before, after))

    def test_accept_keeps_update(self):
        model = _build_model()
        batch = _batch()

        def metric_fn(model, batch):
            # Metric increases with adapter movement away from zero.
            with torch.no_grad():
                gain = 0.0
                for adapter in model.adapter_modules():
                    for parameter in adapter.parameters():
                        gain += float(torch.sum(parameter * parameter))
                return gain

        loop = SafeMetaNetLoop(model=model, metric_fn=metric_fn, learning_rate=0.5)
        first_adapter_before = [parameter.detach().clone() for parameter in model.adapters[0].parameters()]

        result = loop.step(batch)

        self.assertTrue(result.accepted)
        self.assertEqual(result.active_scope, 1)
        self.assertTrue(
            any(not torch.allclose(before, after) for before, after in zip(first_adapter_before, model.adapters[0].parameters()))
        )


if __name__ == "__main__":
    unittest.main()
