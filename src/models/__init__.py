"""Recommendation models and evaluation package."""

from src.models.collaborative import (
    MatrixFactorizationALS,
    NeuralCollaborativeFilter,
)
from src.models.content_based import ContentBasedRecommender
from src.models.evaluation import (
    compute_catalog_coverage,
    compute_mae,
    compute_rmse,
    evaluate_model,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.models.hybrid import HybridRecommender

__all__ = [
    "MatrixFactorizationALS",
    "NeuralCollaborativeFilter",
    "ContentBasedRecommender",
    "HybridRecommender",
    "compute_rmse",
    "compute_mae",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "compute_catalog_coverage",
    "intra_list_diversity",
    "evaluate_model",
]
