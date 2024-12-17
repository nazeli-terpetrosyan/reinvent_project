from typing import List, Literal, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _aggregate(
        all_scores: List[Tuple[np.ndarray, float]],
        mode: Literal["sum", "prod", "harmonic", "chebyshev", "compromise", "max_min", "logarithmic", "exponential"],
        **kwargs
) -> np.ndarray:
    """
    Compute a weighted aggregated score.

    The weights will be normalized.

    :param all_scores: a list of scores and weights
    :param mode: aggregation mode (e.g., sum, prod, harmonic, chebyshev)
    :return: aggregated scores
    """
    sizes = {len(scores) for scores, _ in all_scores}
    if len(sizes) > 1:
        raise ValueError(f"Mismatch in number of scores, got {sizes}")

    scores = np.array([score for score, _ in all_scores])
    weights = np.array([weight for _, weight in all_scores])

    nans = np.isnan(scores)
    if nans.any():
        logger.debug("NaN in component score")

    # Normalize weights to sum to 1
    sum_weights = np.nansum(weights)
    weights = weights / sum_weights if sum_weights != 0 else weights

    # Broadcast weights to scores' shape
    weights = np.array(np.broadcast_to(weights.reshape(-1, 1), scores.shape), dtype=np.float32)
    weights[nans] = np.nan

    if mode == "sum":
        sum_weights = np.nansum(weights, axis=0)
        result = np.nansum(scores * (weights / sum_weights), axis=0)

    elif mode == "prod":
        scores = np.maximum(scores, 1e-8)
        sum_weights = np.maximum(np.nansum(weights, axis=0), 1e-8)
        result = np.nanprod(scores ** (weights / sum_weights), axis=0)

    elif mode == "chebyshev":
        deviations = weights * np.abs(scores - 1)
        result = -np.nanmax(deviations, axis=0)

    elif mode == "compromise":
        p = kwargs.get("p", 2)  # Default p value
        distances = weights * np.abs(scores - 1) ** p
        result = -np.sum(distances, axis=0) ** (1 / p)

    elif mode == "max_min":
        result = np.nanmin(scores / weights, axis=0)

    elif mode == "logarithmic":
        epsilon = kwargs.get("epsilon", 1e-6)
        result = np.sum(weights * np.log(scores + epsilon), axis=0)

    elif mode == "exponential":
        beta = kwargs.get("beta", 1.0)  # Exponential scaling factor
        exp_scores = np.exp(beta * scores)
        result = np.nansum(weights * exp_scores, axis=0)
        print(result)

    else:
        raise ValueError(f"Invalid mode '{mode}'")
    return result


def arithmetic_mean(all_scores: List[Tuple[np.ndarray, float]]) -> np.ndarray:
    """Compute the weighted arithmetic mean."""
    return _aggregate(all_scores, mode="sum")


def geometric_mean(all_scores: List[Tuple[np.ndarray, float]]) -> np.ndarray:
    """Compute the weighted geometric mean."""
    return _aggregate(all_scores, mode="prod")


def chebyshev_scalarization(all_scores: List[Tuple[np.ndarray, float]]) -> np.ndarray:
    """Compute the Chebyshev scalarization."""
    return _aggregate(all_scores, mode="chebyshev")


def compromise_programming_scalarization(all_scores: List[Tuple[np.ndarray, float]], p: float = 2) -> np.ndarray:
    """Compute the Compromise Programming Scalarization."""
    return _aggregate(all_scores, mode="compromise", p=p)


def max_min_scalarization(all_scores: List[Tuple[np.ndarray, float]]) -> np.ndarray:
    """Compute the Max-Min Scalarization."""
    return _aggregate(all_scores, mode="max_min")


def logarithmic_scalarization(all_scores: List[Tuple[np.ndarray, float]], epsilon: float = 1e-6) -> np.ndarray:
    """Compute the Logarithmic Scalarization."""
    return _aggregate(all_scores, mode="logarithmic", epsilon=epsilon)


def exponential_scalarization(all_scores: List[Tuple[np.ndarray, float]], beta: float = 1.0) -> np.ndarray:
    return _aggregate(all_scores, mode="exponential", beta=beta)


def epsilon_constraint_scalarization(all_scores: List[Tuple[np.ndarray, float]]) -> np.ndarray:
    """Compute the Epsilon Constraint Scalarization."""
    scores = np.array([score for score, _ in all_scores])
    constraints_satisfied = np.all(scores[1:] >= 0.2, axis=0)
    return np.where(constraints_satisfied, scores[0], np.zeros_like(scores[0]))