import unittest

from metanet.evaluation_interface import (
    InMemoryModelOutputSource,
    evaluate_predictions,
    export_prediction_dictionary,
)


class EvaluationInterfaceTests(unittest.TestCase):
    def test_export_prediction_dictionary_keeps_mapping_outputs_generic(self) -> None:
        outputs = [{"class": "cat", "score": 0.9}, {"label": "dog", "meta": {"x": 1}}]

        exported = export_prediction_dictionary(outputs)

        self.assertEqual(
            exported,
            {
                "predictions": [
                    {"class": "cat", "score": 0.9},
                    {"label": "dog", "meta": {"x": 1}},
                ]
            },
        )

    def test_export_prediction_dictionary_wraps_non_mapping_outputs(self) -> None:
        exported = export_prediction_dictionary(["raw-output"])

        self.assertEqual(exported, {"predictions": [{"output": "raw-output"}]})

    def test_evaluate_predictions_uses_generic_predictions_collection(self) -> None:
        source = InMemoryModelOutputSource([{"anything": 123}, {"other": True}])

        result = evaluate_predictions(source, lambda predictions: len(predictions))

        self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
