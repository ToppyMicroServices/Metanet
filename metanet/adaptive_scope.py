from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class AdaptiveScopeExpander:
    layers: List[str]
    no_improvement_patience: int
    contribution_scores: Dict[str, float] = field(default_factory=dict)
    current_scope: List[str] = field(init=False)
    best_loss: Optional[float] = field(default=None, init=False)
    no_improvement_steps: int = field(default=0, init=False)
    scope_change_log: List[str] = field(default_factory=list, init=False)
    _layer_priority: Dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("layers must not be empty")
        if self.no_improvement_patience < 1:
            raise ValueError("no_improvement_patience must be >= 1")
        self._layer_priority = {layer: index for index, layer in enumerate(self.layers)}
        self.current_scope = [self.layers[-1]]
        self.scope_change_log.append(
            f"scope_init: editable_scope={self.current_scope}"
        )

    def update_contributions(self, contributions: Dict[str, float]) -> None:
        for layer, score in contributions.items():
            if layer in self.layers:
                self.contribution_scores[layer] = float(score)

    def step(self, loss: float, contributions: Optional[Dict[str, float]] = None) -> bool:
        if contributions:
            self.update_contributions(contributions)
        improved = self.best_loss is None or loss < self.best_loss
        if improved:
            self.best_loss = loss
            self.no_improvement_steps = 0
            return False
        self.no_improvement_steps += 1
        if self.no_improvement_steps >= self.no_improvement_patience:
            self.no_improvement_steps = 0
            return self._expand_scope()
        return False

    def _expand_scope(self) -> bool:
        candidate_layers = [layer for layer in self.layers if layer not in self.current_scope]
        if not candidate_layers:
            return False
        most_influential = max(
            candidate_layers,
            key=lambda layer: (
                self.contribution_scores.get(layer, 0.0),
                self._layer_priority[layer],
            ),
        )
        self.current_scope.append(most_influential)
        self.scope_change_log.append(
            f"scope_expand: added={most_influential}, editable_scope={self.current_scope}"
        )
        return True

    def get_scope(self) -> Iterable[str]:
        return tuple(self.current_scope)
