"""Evaluation metrics module for movie recommendation systems.

Implements standard rating accuracy metrics (RMSE, MAE), top-K ranking metrics
(Precision@K, Recall@K, NDCG@K), catalog coverage, and intra-list diversity.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def compute_rmse(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """Compute Root Mean Squared Error (RMSE).

    Args:
        y_true: Ground truth ratings.
        y_pred: Predicted ratings.

    Returns:
        Scalar RMSE value.
    """
    arr_true = np.asarray(y_true, dtype=np.float64)
    arr_pred = np.asarray(y_pred, dtype=np.float64)
    if len(arr_true) == 0:
        return 0.0
    mse = np.mean((arr_true - arr_pred) ** 2)
    return float(np.sqrt(mse))


def compute_mae(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """Compute Mean Absolute Error (MAE).

    Args:
        y_true: Ground truth ratings.
        y_pred: Predicted ratings.

    Returns:
        Scalar MAE value.
    """
    arr_true = np.asarray(y_true, dtype=np.float64)
    arr_pred = np.asarray(y_pred, dtype=np.float64)
    if len(arr_true) == 0:
        return 0.0
    return float(np.mean(np.abs(arr_true - arr_pred)))


def precision_at_k(
    actual: list[int] | set[int],
    predicted: list[int],
    k: int = 10,
) -> float:
    """Compute Precision@K ranking metric.

    Precision@K = (number of relevant items in top-K) / K

    Args:
        actual: Collection of items considered relevant for the user.
        predicted: Ordered list of recommended item IDs.
        k: Cutoff rank (default: 10).

    Returns:
        Precision score in range [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not predicted or not actual:
        return 0.0

    actual_set = set(actual)
    top_k = predicted[:k]
    hits = sum(1 for item in top_k if item in actual_set)
    return float(hits / k)


def recall_at_k(
    actual: list[int] | set[int],
    predicted: list[int],
    k: int = 10,
) -> float:
    """Compute Recall@K ranking metric.

    Recall@K = (number of relevant items in top-K) / (total relevant items)

    Args:
        actual: Collection of items considered relevant for the user.
        predicted: Ordered list of recommended item IDs.
        k: Cutoff rank (default: 10).

    Returns:
        Recall score in range [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not actual or not predicted:
        return 0.0

    actual_set = set(actual)
    top_k = predicted[:k]
    hits = sum(1 for item in top_k if item in actual_set)
    return float(hits / len(actual_set))


def ndcg_at_k(
    actual: list[int] | set[int],
    predicted: list[int],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K (NDCG@K).

    Uses standard logarithmic discount: 1 / log2(rank + 1).

    Args:
        actual: Collection of items considered relevant for the user.
        predicted: Ordered list of recommended item IDs.
        k: Cutoff rank (default: 10).

    Returns:
        NDCG score in range [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not actual or not predicted:
        return 0.0

    actual_set = set(actual)
    top_k = predicted[:k]

    # Compute Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for idx, item in enumerate(top_k):
        if item in actual_set:
            rank = idx + 1
            dcg += 1.0 / np.log2(rank + 1.0)

    # Compute Ideal Discounted Cumulative Gain (IDCG)
    idcg = 0.0
    max_hits = min(k, len(actual_set))
    for rank in range(1, max_hits + 1):
        idcg += 1.0 / np.log2(rank + 1.0)

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def compute_catalog_coverage(
    all_recommendations: list[list[int]],
    catalog_items: set[int] | list[int] | int,
) -> float:
    """Compute catalog coverage percentage across all user recommendation lists.

    Args:
        all_recommendations: List of recommended item ID lists across users.
        catalog_items: Set/list of all distinct items in catalog, or total count.

    Returns:
        Catalog coverage percentage in range [0.0, 1.0].
    """
    unique_recommended: set[int] = set()
    for rec_list in all_recommendations:
        unique_recommended.update(rec_list)

    if isinstance(catalog_items, int):
        total_items = catalog_items
    else:
        total_items = len(set(catalog_items))

    if total_items <= 0:
        return 0.0

    return float(len(unique_recommended) / total_items)


def intra_list_diversity(
    recommended_items: list[int],
    item_feature_matrix: csr_matrix | np.ndarray,
    movie_id_to_idx: dict[int, int],
) -> float:
    """Compute intra-list diversity as average pairwise cosine distance (1 - similarity).

    Args:
        recommended_items: List of recommended movie IDs.
        item_feature_matrix: Precomputed item feature embeddings (n_items, n_features).
        movie_id_to_idx: Mapping from movie ID to row index in feature matrix.

    Returns:
        Diversity score in range [0.0, 1.0]. Higher implies more diverse recommendations.
    """
    valid_indices = [
        movie_id_to_idx[mid]
        for mid in recommended_items
        if mid in movie_id_to_idx
    ]
    n = len(valid_indices)
    if n <= 1:
        return 0.0

    features = item_feature_matrix[valid_indices]
    if issparse(features):
        features = features.toarray()

    sim_matrix = cosine_similarity(features)
    # Average pairwise distance over all off-diagonal pairs
    triu_indices = np.triu_indices(n, k=1)
    pairwise_distances = 1.0 - sim_matrix[triu_indices]
    # Bound distances between 0 and 1
    pairwise_distances = np.clip(pairwise_distances, 0.0, 1.0)
    return float(np.mean(pairwise_distances))


def evaluate_model(
    model: Any,
    test_data: pd.DataFrame,
    train_data: pd.DataFrame | None = None,
    k: int = 10,
    relevance_threshold: float = 3.5,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    max_eval_users: int = 200,
) -> dict[str, float]:
    """Evaluate a recommendation model comprehensively on test data.

    Evaluates:
    - Rating error: RMSE and MAE across test interactions.
    - Top-K ranking: Precision@K, Recall@K, and NDCG@K.
    - Catalog Coverage: Percentage of catalog items recommended.

    Args:
        model: Recommendation model implementing `predict(user_id, item_id)`
            and optionally `recommend(user_id, n=k)`.
        test_data: Test ratings DataFrame.
        train_data: Optional train ratings DataFrame (used to exclude train items).
        k: Rank cutoff for top-K metrics.
        relevance_threshold: Rating cutoff for defining positive user interest.
        user_col: User ID column name.
        item_col: Item ID column name.
        rating_col: Rating column name.
        max_eval_users: Maximum number of users to evaluate ranking for.

    Returns:
        Dictionary of computed evaluation metrics.
    """
    logger.info("Evaluating model on %d test interactions...", len(test_data))

    # 1. Rating Prediction Error (RMSE & MAE)
    y_true: list[float] = []
    y_pred: list[float] = []

    has_predict = hasattr(model, "predict") and callable(getattr(model, "predict"))
    if has_predict:
        for _, row in test_data.iterrows():
            uid = int(row[user_col])
            iid = int(row[item_col])
            true_r = float(row[rating_col])
            try:
                pred_r = float(model.predict(uid, iid))
            except Exception:
                pred_r = 3.5
            y_true.append(true_r)
            y_pred.append(pred_r)

    rmse = compute_rmse(y_true, y_pred) if y_true else float("nan")
    mae = compute_mae(y_true, y_pred) if y_true else float("nan")

    # 2. Ranking Metrics (Precision@K, Recall@K, NDCG@K)
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    all_recommendations: list[list[int]] = []

    catalog_items = set(test_data[item_col].unique())
    if train_data is not None:
        catalog_items.update(train_data[item_col].unique())

    has_recommend = hasattr(model, "recommend") and callable(getattr(model, "recommend"))

    if has_recommend:
        # Group relevant items per user in test set
        test_grouped = test_data[test_data[rating_col] >= relevance_threshold].groupby(user_col)[item_col].apply(set)
        test_users = list(test_grouped.index)

        if max_eval_users and len(test_users) > max_eval_users:
            np.random.seed(42)
            eval_users = list(np.random.choice(test_users, size=max_eval_users, replace=False))
        else:
            eval_users = test_users

        for uid in eval_users:
            relevant_items = test_grouped.get(uid, set())
            if not relevant_items:
                continue

            try:
                recs_output = model.recommend(uid, n=k)
                if not recs_output:
                    continue

                if isinstance(recs_output[0], dict):
                    rec_ids = [int(r["movie_id"]) for r in recs_output]
                elif isinstance(recs_output[0], tuple):
                    rec_ids = [int(r[0]) for r in recs_output]
                else:
                    rec_ids = [int(r) for r in recs_output]
            except Exception as exc:
                logger.debug("Failed recommendation for user %d during eval: %s", uid, exc)
                continue

            all_recommendations.append(rec_ids)
            p_k = precision_at_k(relevant_items, rec_ids, k=k)
            r_k = recall_at_k(relevant_items, rec_ids, k=k)
            n_k = ndcg_at_k(relevant_items, rec_ids, k=k)

            precisions.append(p_k)
            recalls.append(r_k)
            ndcgs.append(n_k)

    avg_precision = float(np.mean(precisions)) if precisions else 0.0
    avg_recall = float(np.mean(recalls)) if recalls else 0.0
    avg_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    coverage = compute_catalog_coverage(all_recommendations, catalog_items) if all_recommendations else 0.0

    metrics = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        f"precision@{k}": round(avg_precision, 4),
        f"recall@{k}": round(avg_recall, 4),
        f"ndcg@{k}": round(avg_ndcg, 4),
        "catalog_coverage": round(coverage, 4),
        "evaluated_test_interactions": len(y_true),
        "evaluated_ranking_users": len(precisions),
    }

    logger.info("Evaluation results: %s", metrics)
    return metrics
