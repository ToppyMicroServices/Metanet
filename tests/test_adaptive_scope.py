import unittest

from metanet import AdaptiveScopeExpander


class AdaptiveScopeExpanderTests(unittest.TestCase):
    def test_starts_with_last_layer_only(self) -> None:
        expander = AdaptiveScopeExpander(
            layers=["encoder", "mid_adapter", "last_adapter"],
            no_improvement_patience=2,
        )

        self.assertEqual(tuple(expander.get_scope()), ("last_adapter",))
        self.assertEqual(
            expander.scope_change_log,
            ["scope_init: editable_scope=['last_adapter']"],
        )

    def test_expands_after_patience_using_most_influential_layer(self) -> None:
        expander = AdaptiveScopeExpander(
            layers=["encoder", "mid_adapter", "last_adapter"],
            no_improvement_patience=2,
        )

        expander.step(loss=1.0, contributions={"encoder": 0.1, "mid_adapter": 0.9})
        expander.step(loss=1.0)
        expanded = expander.step(loss=1.0)

        self.assertTrue(expanded)
        self.assertEqual(
            tuple(expander.get_scope()),
            ("last_adapter", "mid_adapter"),
        )
        self.assertIn(
            "scope_expand: added=mid_adapter, editable_scope=['last_adapter', 'mid_adapter']",
            expander.scope_change_log,
        )

    def test_deterministic_tie_break_uses_layer_order(self) -> None:
        expander = AdaptiveScopeExpander(
            layers=["encoder", "mid_adapter", "last_adapter"],
            no_improvement_patience=1,
        )

        expander.step(loss=1.0, contributions={"encoder": 0.5, "mid_adapter": 0.5})
        expander.step(loss=1.0)

        self.assertEqual(
            tuple(expander.get_scope()),
            ("last_adapter", "mid_adapter"),
        )


if __name__ == "__main__":
    unittest.main()
