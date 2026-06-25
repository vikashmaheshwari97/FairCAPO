from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from heal_capo.core import EvaluationResult


ObjectiveName = str


DEFAULT_OBJECTIVES: tuple[ObjectiveName, ...] = (
    "performance",
    "cost",
    "risk",
    "fairness_risk",
)


@dataclass(frozen=True)
class ObjectiveSpec:
    """
    Defines objective orientation.

    maximize:
      - higher is better

    minimize:
      - lower is better
    """

    name: ObjectiveName
    direction: str

    def is_maximize(self) -> bool:
        return self.direction == "maximize"

    def is_minimize(self) -> bool:
        return self.direction == "minimize"


DEFAULT_OBJECTIVE_SPECS: tuple[ObjectiveSpec, ...] = (
    ObjectiveSpec("performance", "maximize"),
    ObjectiveSpec("cost", "minimize"),
    ObjectiveSpec("risk", "minimize"),
    ObjectiveSpec("fairness_risk", "minimize"),
)


@dataclass
class NormalizationBounds:
    """
    Bounds used to normalize objectives to [0, 1] utility space.

    In normalized utility space:
      - 1.0 is best
      - 0.0 is worst
    """

    minimum: dict[str, float]
    maximum: dict[str, float]


@dataclass
class MOMetricSummary:
    num_points: int
    num_objectives: int
    hypervolume: float
    optimistic_hypervolume: float
    pessimistic_hypervolume: float
    approximation_gap: float
    nr2: float
    bounds: dict
    objective_names: list[str]

    def to_dict(self) -> dict:
        return {
            "num_points": self.num_points,
            "num_objectives": self.num_objectives,
            "hypervolume": self.hypervolume,
            "optimistic_hypervolume": self.optimistic_hypervolume,
            "pessimistic_hypervolume": self.pessimistic_hypervolume,
            "approximation_gap": self.approximation_gap,
            "nr2": self.nr2,
            "bounds": self.bounds,
            "objective_names": self.objective_names,
        }


def get_objective_value(
    result: EvaluationResult,
    objective: str,
) -> float:
    value = getattr(result, objective, None)

    if value is None:
        value = result.details.get(objective, 0.0) if result.details else 0.0

    try:
        value = float(value)
    except Exception:
        return 0.0

    if math.isnan(value) or math.isinf(value):
        return 0.0

    return value


def results_to_matrix(
    results: Iterable[EvaluationResult],
    objective_specs: Sequence[ObjectiveSpec] = DEFAULT_OBJECTIVE_SPECS,
) -> list[list[float]]:
    return [
        [get_objective_value(result, spec.name) for spec in objective_specs]
        for result in results
    ]


def infer_bounds(
    results: Iterable[EvaluationResult],
    objective_specs: Sequence[ObjectiveSpec] = DEFAULT_OBJECTIVE_SPECS,
    padding: float = 1e-12,
) -> NormalizationBounds:
    vals = list(results)

    if not vals:
        return NormalizationBounds(
            minimum={spec.name: 0.0 for spec in objective_specs},
            maximum={spec.name: 1.0 for spec in objective_specs},
        )

    minimum = {}
    maximum = {}

    for spec in objective_specs:
        values = [get_objective_value(result, spec.name) for result in vals]
        lo = min(values)
        hi = max(values)

        if abs(hi - lo) <= padding:
            hi = lo + 1.0

        minimum[spec.name] = lo
        maximum[spec.name] = hi

    return NormalizationBounds(minimum=minimum, maximum=maximum)

def fixed_bounds_from_config(
    bounds_config: dict | None,
    objective_specs: Sequence[ObjectiveSpec] = DEFAULT_OBJECTIVE_SPECS,
) -> NormalizationBounds | None:
    """
    Build NormalizationBounds from config.

    Expected config format:
      bounds:
        performance: [0.0, 1.0]
        cost: [0.0, 50.0]
        risk: [0.0, 1.0]
        fairness_risk: [0.0, 1.0]

    Returns None if bounds_config is empty.
    """
    if not bounds_config:
        return None

    minimum = {}
    maximum = {}

    for spec in objective_specs:
        raw = bounds_config.get(spec.name)

        if raw is None:
            return None

        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(
                f"Bounds for objective '{spec.name}' must be a two-item list: [min, max]."
            )

        lo = float(raw[0])
        hi = float(raw[1])

        if hi <= lo:
            raise ValueError(
                f"Invalid bounds for objective '{spec.name}': max must be greater than min."
            )

        minimum[spec.name] = lo
        maximum[spec.name] = hi

    return NormalizationBounds(
        minimum=minimum,
        maximum=maximum,
    )


def normalize_result(
    result: EvaluationResult,
    bounds: NormalizationBounds,
    objective_specs: Sequence[ObjectiveSpec] = DEFAULT_OBJECTIVE_SPECS,
) -> list[float]:
    """
    Convert raw objectives into utility values in [0, 1].

    For maximization objectives:
      utility = (value - min) / (max - min)

    For minimization objectives:
      utility = (max - value) / (max - min)
    """
    vector = []

    for spec in objective_specs:
        value = get_objective_value(result, spec.name)
        lo = bounds.minimum[spec.name]
        hi = bounds.maximum[spec.name]
        denom = hi - lo

        if denom <= 0:
            utility = 1.0
        elif spec.is_maximize():
            utility = (value - lo) / denom
        else:
            utility = (hi - value) / denom

        utility = min(1.0, max(0.0, utility))
        vector.append(utility)

    return vector


def normalize_results(
    results: Iterable[EvaluationResult],
    bounds: NormalizationBounds | None = None,
    objective_specs: Sequence[ObjectiveSpec] = DEFAULT_OBJECTIVE_SPECS,
) -> list[list[float]]:
    vals = list(results)

    if bounds is None:
        bounds = infer_bounds(vals, objective_specs)

    return [
        normalize_result(result, bounds, objective_specs)
        for result in vals
    ]


def dominates_normalized(
    left: Sequence[float],
    right: Sequence[float],
) -> bool:
    """
    Pareto dominance in normalized utility space.

    All objectives are maximized after normalization.
    """
    return all(l >= r for l, r in zip(left, right)) and any(
        l > r for l, r in zip(left, right)
    )


def nondominated_points(points: Iterable[Sequence[float]]) -> list[list[float]]:
    point_list = [list(point) for point in points]
    keep = []

    for idx, point in enumerate(point_list):
        dominated = False

        for other_idx, other in enumerate(point_list):
            if idx == other_idx:
                continue

            if dominates_normalized(other, point):
                dominated = True
                break

        if not dominated:
            keep.append(point)

    return keep


def _hypervolume_recursive(
    points: list[list[float]],
    reference: list[float],
) -> float:
    """
    Exact hypervolume for small point sets in normalized utility space.

    Assumes all objectives are maximized and reference is usually [0, ..., 0].

    This implementation uses inclusion-exclusion over axis-aligned boxes.
    It is exponential in the number of points, so it is intended for small
    portfolios and tests. That is fine for our current prompt portfolios.
    """
    if not points:
        return 0.0

    dimension = len(reference)
    hv = 0.0

    for size in range(1, len(points) + 1):
        sign = 1.0 if size % 2 == 1 else -1.0

        for subset in itertools.combinations(points, size):
            upper = [
                min(point[d] for point in subset)
                for d in range(dimension)
            ]

            volume = 1.0

            for d in range(dimension):
                edge = max(0.0, upper[d] - reference[d])
                volume *= edge

            hv += sign * volume

    return max(0.0, hv)


def hypervolume(
    results: Iterable[EvaluationResult],
    bounds: NormalizationBounds | None = None,
    objective_specs: Sequence[ObjectiveSpec] = DEFAULT_OBJECTIVE_SPECS,
    reference: Sequence[float] | None = None,
) -> float:
    vals = list(results)

    if not vals:
        return 0.0

    if bounds is None:
        bounds = infer_bounds(vals, objective_specs)

    points = normalize_results(vals, bounds, objective_specs)
    points = nondominated_points(points)

    ref = list(reference) if reference is not None else [0.0] * len(objective_specs)

    return _hypervolume_recursive(points, ref)


def optimistic_bounds(
    candidate_results: Iterable[EvaluationResult],
    reference_results: Iterable[EvaluationResult] | None = None,
    objective_specs: Sequence[ObjectiveSpec] = DEFAULT_OBJECTIVE_SPECS,
) -> NormalizationBounds:
    """
    Optimistic bounds use candidate + reference points when available.

    If no reference is available, this is equivalent to inferred candidate bounds.
    """
    vals = list(candidate_results)
    refs = list(reference_results or [])

    return infer_bounds(vals + refs, objective_specs)


def pessimistic_bounds(
    candidate_results: Iterable[EvaluationResult],
    objective_specs: Sequence[ObjectiveSpec] = DEFAULT_OBJECTIVE_SPECS,
) -> NormalizationBounds:
    """
    Pessimistic bounds are inferred only from the evaluated candidate set.

    This usually produces a local-view hypervolume.
    """
    return infer_bounds(candidate_results, objective_specs)


def optimistic_hypervolume(
    candidate_results: Iterable[EvaluationResult],
    reference_results: Iterable[EvaluationResult] | None = None,
    objective_specs: Sequence[ObjectiveSpec] = DEFAULT_OBJECTIVE_SPECS,
) -> float:
    vals = list(candidate_results)
    bounds = optimistic_bounds(vals, reference_results, objective_specs)

    return hypervolume(vals, bounds, objective_specs)


def pessimistic_hypervolume(
    candidate_results: Iterable[EvaluationResult],
    objective_specs: Sequence[ObjectiveSpec] = DEFAULT_OBJECTIVE_SPECS,
) -> float:
    vals = list(candidate_results)
    bounds = pessimistic_bounds(vals, objective_specs)

    return hypervolume(vals, bounds, objective_specs)


def chebychev_utility(
    point: Sequence[float],
    weights: Sequence[float],
    ideal: Sequence[float] | None = None,
) -> float:
    """
    Chebychev utility in normalized maximization space.

    Lower Chebychev distance from ideal is better, so utility is:
      1 - max_i w_i * |ideal_i - point_i|
    """
    if ideal is None:
        ideal = [1.0] * len(point)

    distance = max(
        weights[i] * abs(ideal[i] - point[i])
        for i in range(len(point))
    )

    return 1.0 - distance


def sample_preference_vectors(
    num_vectors: int,
    num_objectives: int,
    seed: int = 0,
) -> list[list[float]]:
    """
    Sample random simplex preference vectors.

    For paper-level nR2, use num_vectors=500.
    For local smoke tests, 50 is enough.
    """
    if num_vectors <= 0:
        return []

    rng = random.Random(seed)
    vectors = []

    for _ in range(num_vectors):
        raw = [rng.random() for _ in range(num_objectives)]
        total = sum(raw)

        if total <= 0:
            vectors.append([1.0 / num_objectives] * num_objectives)
        else:
            vectors.append([value / total for value in raw])

    return vectors


def nr2(
    candidate_results: Iterable[EvaluationResult],
    reference_results: Iterable[EvaluationResult] | None = None,
    objective_specs: Sequence[ObjectiveSpec] = DEFAULT_OBJECTIVE_SPECS,
    num_preference_vectors: int = 500,
    seed: int = 0,
) -> float:
    """
    Normalized R2-style score using Chebychev utilities.

    This returns a normalized regret/gap-like value:
      0.0 is best
      larger is worse

    If reference_results are provided, we compare candidates against that
    reference set. If not, we use the candidate set as its own reference and
    the result is 0.0.
    """
    candidates = list(candidate_results)
    references = list(reference_results or candidates)

    if not candidates:
        return 1.0

    bounds = infer_bounds(candidates + references, objective_specs)

    candidate_points = normalize_results(candidates, bounds, objective_specs)
    reference_points = normalize_results(references, bounds, objective_specs)

    vectors = sample_preference_vectors(
        num_vectors=num_preference_vectors,
        num_objectives=len(objective_specs),
        seed=seed,
    )

    if not vectors:
        return 0.0

    regrets = []

    for weights in vectors:
        best_candidate = max(
            chebychev_utility(point, weights)
            for point in candidate_points
        )
        best_reference = max(
            chebychev_utility(point, weights)
            for point in reference_points
        )

        regrets.append(max(0.0, best_reference - best_candidate))

    return sum(regrets) / len(regrets)


def approximation_gap(
    candidate_results: Iterable[EvaluationResult],
    reference_results: Iterable[EvaluationResult],
    objective_specs: Sequence[ObjectiveSpec] = DEFAULT_OBJECTIVE_SPECS,
) -> float:
    """
    Average distance from each reference point to the nearest candidate point.

    Computed in normalized utility space. Lower is better.
    """
    candidates = list(candidate_results)
    references = list(reference_results)

    if not candidates and not references:
        return 0.0

    if not candidates:
        return 1.0

    if not references:
        return 0.0

    bounds = infer_bounds(candidates + references, objective_specs)
    candidate_points = normalize_results(candidates, bounds, objective_specs)
    reference_points = normalize_results(references, bounds, objective_specs)

    distances = []

    for ref in reference_points:
        nearest = min(
            math.dist(ref, candidate)
            for candidate in candidate_points
        )
        distances.append(nearest)

    return sum(distances) / len(distances)

def nr2_with_bounds(
    candidate_results: Iterable[EvaluationResult],
    reference_results: Iterable[EvaluationResult] | None = None,
    objective_specs: Sequence[ObjectiveSpec] = DEFAULT_OBJECTIVE_SPECS,
    num_preference_vectors: int = 500,
    seed: int = 0,
    bounds: NormalizationBounds | None = None,
) -> float:
    candidates = list(candidate_results)
    references = list(reference_results or candidates)

    if not candidates:
        return 1.0

    if bounds is None:
        bounds = infer_bounds(candidates + references, objective_specs)

    candidate_points = normalize_results(candidates, bounds, objective_specs)
    reference_points = normalize_results(references, bounds, objective_specs)

    vectors = sample_preference_vectors(
        num_vectors=num_preference_vectors,
        num_objectives=len(objective_specs),
        seed=seed,
    )

    if not vectors:
        return 0.0

    regrets = []

    for weights in vectors:
        best_candidate = max(
            chebychev_utility(point, weights)
            for point in candidate_points
        )
        best_reference = max(
            chebychev_utility(point, weights)
            for point in reference_points
        )

        regrets.append(max(0.0, best_reference - best_candidate))

    return sum(regrets) / len(regrets)


def approximation_gap_with_bounds(
    candidate_results: Iterable[EvaluationResult],
    reference_results: Iterable[EvaluationResult],
    objective_specs: Sequence[ObjectiveSpec] = DEFAULT_OBJECTIVE_SPECS,
    bounds: NormalizationBounds | None = None,
) -> float:
    candidates = list(candidate_results)
    references = list(reference_results)

    if not candidates and not references:
        return 0.0

    if not candidates:
        return 1.0

    if not references:
        return 0.0

    if bounds is None:
        bounds = infer_bounds(candidates + references, objective_specs)

    candidate_points = normalize_results(candidates, bounds, objective_specs)
    reference_points = normalize_results(references, bounds, objective_specs)

    distances = []

    for ref in reference_points:
        nearest = min(
            math.dist(ref, candidate)
            for candidate in candidate_points
        )
        distances.append(nearest)

    return sum(distances) / len(distances)

def summarize_mo_metrics(
    candidate_results: Iterable[EvaluationResult],
    reference_results: Iterable[EvaluationResult] | None = None,
    objective_specs: Sequence[ObjectiveSpec] = DEFAULT_OBJECTIVE_SPECS,
    num_preference_vectors: int = 500,
    seed: int = 0,
    bounds: NormalizationBounds | None = None,
) -> MOMetricSummary:
    candidates = list(candidate_results)
    references = list(reference_results or candidates)

    using_fixed_bounds = bounds is not None

    if bounds is None:
        bounds = infer_bounds(candidates + references, objective_specs)

    hv = hypervolume(candidates, bounds, objective_specs)

    if using_fixed_bounds:
        hv_opt = hypervolume(candidates, bounds, objective_specs)
        hv_pes = hypervolume(candidates, bounds, objective_specs)
    else:
        hv_opt = optimistic_hypervolume(candidates, references, objective_specs)
        hv_pes = pessimistic_hypervolume(candidates, objective_specs)

    gap = approximation_gap_with_bounds(
        candidate_results=candidates,
        reference_results=references,
        objective_specs=objective_specs,
        bounds=bounds,
    )

    r2 = nr2_with_bounds(
        candidate_results=candidates,
        reference_results=references,
        objective_specs=objective_specs,
        num_preference_vectors=num_preference_vectors,
        seed=seed,
        bounds=bounds,
    )

    return MOMetricSummary(
        num_points=len(candidates),
        num_objectives=len(objective_specs),
        hypervolume=hv,
        optimistic_hypervolume=hv_opt,
        pessimistic_hypervolume=hv_pes,
        approximation_gap=gap,
        nr2=r2,
        bounds={
            "minimum": bounds.minimum,
            "maximum": bounds.maximum,
            "fixed": using_fixed_bounds,
        },
        objective_names=[spec.name for spec in objective_specs],
    )