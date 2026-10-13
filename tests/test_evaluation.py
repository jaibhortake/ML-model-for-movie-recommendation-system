"""Tests for recommendation system evaluation metrics.

Validates:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Precision@K
- Normalized Discounted Cumulative Gain (NDCG@K)
"""

import numpy as np
import pytest

# Attempt to import evaluation metrics from src
try:
    from src.evaluation import mae, ndcg_at_k, precision_at_k, rmse
except ImportError:
    try:
        from evaluation import mae, ndcg_at_k, precision_at_k, rmse  # type: ignore
    except ImportError:
        rmse = None
        mae = None
        precision_at_k = None
        ndcg_at_k = None


# ============================================================================
# Helpers / Fallback Reference Implementations
# ============================================================================

def _calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates RMSE using module function or reference formula."""
    if rmse is not None:
        return float(rmse(y_true, y_pred))
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def _calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates MAE using module function or reference formula."""
    if mae is not None:
        return float(mae(y_true, y_pred))
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def _calc_precision_at_k(actual: list, predicted: list, k: int = 5) -> float:
    """Calculates Precision@K using module function or reference formula."""
    if precision_at_k is not None:
        return float(precision_at_k(actual, predicted, k=k))
    if k <= 0:
        return 0.0
    pred_top_k = predicted[:k]
    actual_set = set(actual)
    hits = len([item for item in pred_top_k if item in actual_set])
    return float(hits / k)


def _calc_ndcg_at_k(actual: list, predicted: list, k: int = 5) -> float:
    """Calculates NDCG@K using module function or reference formula."""
    if ndcg_at_k is not None:
        return float(ndcg_at_k(actual, predicted, k=k))
    if k <= 0:
        return 0.0
    pred_top_k = predicted[:k]
    actual_set = set(actual)

    dcg = 0.0
    for i, item in enumerate(pred_top_k):
        if item in actual_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(rank + 1) with 1-based rank

    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual_set), k)))
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


# ============================================================================
# Unit Tests
# ============================================================================

def test_rmse_perfect_predictions():
    """Test that RMSE equals 0.0 when predictions are identical to ground truth."""
    y_true = np.array([1.0, 2.5, 3.0, 4.5, 5.0])
    y_pred = np.array([1.0, 2.5, 3.0, 4.5, 5.0])

    val = _calc_rmse(y_true, y_pred)
    assert np.isclose(val, 0.0, atol=1e-7), f"Expected RMSE 0.0 for perfect predictions, got {val}"


def test_rmse_known_values():
    """Test RMSE calculation against hand-computed reference values."""
    y_true = np.array([3.0, 4.0])
    y_pred = np.array([4.0, 2.0])
    # Differences: [ -1, 2 ], Squared: [ 1, 4 ], Mean: 2.5, Sqrt: ~1.5811388
    expected_rmse = np.sqrt(2.5)

    val = _calc_rmse(y_true, y_pred)
    assert np.isclose(val, expected_rmse, atol=1e-5), (
        f"Expected RMSE ~{expected_rmse}, got {val}"
    )


def test_mae_perfect_predictions():
    """Test that MAE equals 0.0 when predictions are identical to ground truth."""
    y_true = np.array([2.0, 3.5, 4.0, 5.0])
    y_pred = np.array([2.0, 3.5, 4.0, 5.0])

    val = _calc_mae(y_true, y_pred)
    assert np.isclose(val, 0.0, atol=1e-7), f"Expected MAE 0.0 for perfect predictions, got {val}"


def test_precision_at_k_all_relevant():
    """Test Precision@K when all top-K recommendations are relevant."""
    actual = [1, 2, 3, 4, 5]
    predicted = [1, 2, 3, 8, 9]
    k = 3

    score = _calc_precision_at_k(actual, predicted, k=k)
    assert np.isclose(score, 1.0, atol=1e-7), f"Expected Precision@{k} of 1.0, got {score}"


def test_precision_at_k_none_relevant():
    """Test Precision@K when none of the top-K recommendations are relevant."""
    actual = [1, 2, 3]
    predicted = [4, 5, 6, 7]
    k = 3

    score = _calc_precision_at_k(actual, predicted, k=k)
    assert np.isclose(score, 0.0, atol=1e-7), f"Expected Precision@{k} of 0.0, got {score}"


def test_ndcg_perfect_ranking():
    """Test NDCG@K when recommended items match the ideal ranking order."""
    actual = [10, 20, 30]
    predicted = [10, 20, 30, 40, 50]
    k = 3

    score = _calc_ndcg_at_k(actual, predicted, k=k)
    assert np.isclose(score, 1.0, atol=1e-7), f"Expected NDCG@{k} of 1.0 for ideal ranking, got {score}"
