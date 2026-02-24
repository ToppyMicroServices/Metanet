import unittest

from metanet import AblationConfig, AdaptationConfig, safe_metanet_adaptation_loop


class SafeMetaNetLoopTests(unittest.TestCase):
    def test_accepts_improving_update(self) -> None:
        config = AdaptationConfig(steps=1, initial_scope=1, max_scope=3, safety_margin=0.1)

        result = safe_metanet_adaptation_loop(
            initial_state={"x": 1.0},
            propose_update=lambda state, scope: {"x": state["x"] + scope},
            evaluate=lambda state: float(state["x"]),
            config=config,
        )

        self.assertEqual(result.final_state["x"], 2.0)
        self.assertEqual(result.final_scope, 1)
        self.assertTrue(result.events[0].accepted)

    def test_rolls_back_and_expands_scope_on_reject(self) -> None:
        config = AdaptationConfig(steps=1, initial_scope=1, max_scope=3, scope_increment=1, safety_margin=0.5)

        result = safe_metanet_adaptation_loop(
            initial_state={"x": 10.0},
            propose_update=lambda state, scope: {"x": state["x"] - 1.0},
            evaluate=lambda state: float(state["x"]),
            config=config,
        )

        self.assertEqual(result.final_state["x"], 10.0)
        self.assertEqual(result.final_scope, 2)
        self.assertFalse(result.events[0].accepted)
        self.assertEqual(result.events[0].rejection_reason, "insufficient_improvement")

    def test_ablation_can_disable_rollback(self) -> None:
        config = AdaptationConfig(
            steps=1,
            initial_scope=1,
            max_scope=1,
            safety_margin=0.5,
            ablations=AblationConfig(disable_rollback=True),
        )

        result = safe_metanet_adaptation_loop(
            initial_state={"x": 10.0},
            propose_update=lambda state, scope: {"x": state["x"] - 1.0},
            evaluate=lambda state: float(state["x"]),
            config=config,
        )

        self.assertEqual(result.final_state["x"], 9.0)
        self.assertFalse(result.events[0].accepted)

    def test_ablation_can_disable_safety_checks(self) -> None:
        config = AdaptationConfig(
            steps=1,
            initial_scope=1,
            max_scope=1,
            safety_margin=100.0,
            ablations=AblationConfig(disable_safety_checks=True),
        )

        result = safe_metanet_adaptation_loop(
            initial_state={"x": 10.0},
            propose_update=lambda state, scope: {"x": state["x"] - 1.0},
            evaluate=lambda state: float(state["x"]),
            config=config,
        )

        self.assertEqual(result.final_state["x"], 9.0)
        self.assertTrue(result.events[0].accepted)


if __name__ == "__main__":
    unittest.main()
