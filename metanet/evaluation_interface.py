from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


Prediction = dict[str, Any]
PredictionDictionary = dict[str, list[Prediction]]


class PredictionSource(Protocol):
    """Provides predictions in a tool-agnostic dictionary format."""

    def prediction_dictionary(self) -> PredictionDictionary:
        ...


def export_prediction_dictionary(model_outputs: Iterable[Any]) -> PredictionDictionary:
    """Convert model outputs to a generic prediction dictionary.

    The output format is intentionally minimal so it can be swapped with
    external predictions.json producers (e.g. YOLOZU) later.
    """

    predictions: list[Prediction] = []
    for output in model_outputs:
        if isinstance(output, Mapping):
            predictions.append(dict(output))
        else:
            predictions.append({"output": output})

    return {"predictions": predictions}


class InMemoryModelOutputSource:
    def __init__(self, model_outputs: Iterable[Any]) -> None:
        self._model_outputs = model_outputs

    def prediction_dictionary(self) -> PredictionDictionary:
        return export_prediction_dictionary(self._model_outputs)


def evaluate_predictions(
    source: PredictionSource,
    evaluator: Callable[[Sequence[Mapping[str, Any]]], Any],
) -> Any:
    """Evaluate predictions without imposing dataset-specific assumptions.

    Missing or malformed sources are treated as an empty prediction list.
    """

    prediction_dictionary = source.prediction_dictionary()
    predictions = prediction_dictionary.get("predictions", [])
    return evaluator(predictions)
