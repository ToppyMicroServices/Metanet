from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Callable, Dict, List, Mapping, MutableMapping


State = MutableMapping[str, Any]
StateFactory = Callable[[State, int], Mapping[str, Any]]
Evaluator = Callable[[Mapping[str, Any]], float]


@dataclass(frozen=True)
class AblationConfig:
    disable_safety_checks: bool = False
    disable_scope_expansion: bool = False
    disable_rollback: bool = False


@dataclass(frozen=True)
class AdaptationConfig:
    steps: int = 1
    initial_scope: int = 1
    max_scope: int = 1
    scope_increment: int = 1
    safety_margin: float = 0.0
    ablations: AblationConfig = field(default_factory=AblationConfig)


@dataclass(frozen=True)
class AdaptationEvent:
    step: int
    scope: int
    accepted: bool
    score_before: float
    score_after: float
    rejection_reason: str | None = None


@dataclass(frozen=True)
class AdaptationResult:
    final_state: Dict[str, Any]
    final_score: float
    final_scope: int
    events: List[AdaptationEvent]


def _as_finite_score(value: float) -> float:
    score = float(value)
    if not isfinite(score):
        raise ValueError("Evaluator must return a finite score.")
    return score


def _is_state_safe(candidate: Mapping[str, Any]) -> bool:
    for value in candidate.values():
        if isinstance(value, (int, float)) and not isfinite(float(value)):
            return False
    return True


def safe_metanet_adaptation_loop(
    initial_state: Mapping[str, Any],
    propose_update: StateFactory,
    evaluate: Evaluator,
    config: AdaptationConfig,
) -> AdaptationResult:
    if config.steps < 0:
        raise ValueError("steps must be >= 0")
    if config.initial_scope < 1:
        raise ValueError("initial_scope must be >= 1")
    if config.max_scope < config.initial_scope:
        raise ValueError("max_scope must be >= initial_scope")
    if config.scope_increment < 1:
        raise ValueError("scope_increment must be >= 1")

    current_state: Dict[str, Any] = dict(initial_state)
    current_scope = config.initial_scope
    current_score = _as_finite_score(evaluate(current_state))
    events: List[AdaptationEvent] = []

    for step in range(config.steps):
        score_before = current_score
        candidate_state = dict(propose_update(dict(current_state), current_scope))
        candidate_score = _as_finite_score(evaluate(candidate_state))

        rejection_reason = None
        accepted = True

        if not config.ablations.disable_safety_checks:
            if not _is_state_safe(candidate_state):
                accepted = False
                rejection_reason = "unsafe_state"
            elif candidate_score < current_score + config.safety_margin:
                accepted = False
                rejection_reason = "insufficient_improvement"

        if accepted:
            current_state = candidate_state
            current_score = candidate_score
        else:
            if not config.ablations.disable_scope_expansion and current_scope < config.max_scope:
                current_scope = min(config.max_scope, current_scope + config.scope_increment)

            if config.ablations.disable_rollback:
                current_state = candidate_state
                current_score = candidate_score
                accepted = True
                rejection_reason = "rollback_disabled"

        events.append(
            AdaptationEvent(
                step=step,
                scope=current_scope,
                accepted=accepted,
                score_before=score_before,
                score_after=candidate_score,
                rejection_reason=rejection_reason,
            )
        )

    return AdaptationResult(
        final_state=current_state,
        final_score=current_score,
        final_scope=current_scope,
        events=events,
    )
