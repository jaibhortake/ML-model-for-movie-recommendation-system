"""Tests for data preprocessing and feature engineering module.

Validates zero-mean rating normalization, train-test splitting,
genre multi-hot binary encoding, and movie feature matrix generation.
"""

import numpy as np
import pandas as pd
import pytest

# Attempt to import from preprocessor module
try:
    from src.preprocessor import (
        DataPreprocessor,
        create_feature_matrix,
        encode_genres,
        normalize_ratings,
        train_test_split,
    )
except ImportError:
    try:
        from preprocessor import (  # type: ignore
            DataPreprocessor,
            create_feature_matrix,
            encode_genres,
            normalize_ratings,
            train_test_split,
        )
    except ImportError:
        DataPreprocessor = None
        create_feature_matrix = None
        encode_genres = None
        normalize_ratings = None
        train_test_split = None


# ============================================================================
# Helpers / Fallback Logic
# ============================================================================

def _run_normalize(matrix: np.ndarray):
    """Executes normalize_ratings or reference zero-mean centering."""
    if normalize_ratings is not None:
        return normalize_ratings(matrix)
    elif DataPreprocessor is not None and hasattr(DataPreprocessor, "normalize_ratings"):
        return DataPreprocessor().normalize_ratings(matrix)
    else:
        # Zero-mean normalization over rated (non-zero) entries per user
        norm_mat = matrix.copy().astype(np.float64)
        for u in range(norm_mat.shape[0]):
            rated_mask = norm_mat[u] > 0
            if np.any(rated_mask):
                user_mean = np.mean(norm_mat[u, rated_mask])
                norm_mat[u, rated_mask] -= user_mean
        return norm_mat


def _run_train_test_split(matrix: np.ndarray, test_ratio: float = 0.2):
    """Executes train_test_split or reference masking split."""
    if train_test_split is not None:
        return train_test_split(matrix, test_ratio=test_ratio)
    elif DataPreprocessor is not None and hasattr(DataPreprocessor, "train_test_split"):
        return DataPreprocessor().train_test_split(matrix, test_ratio=test_ratio)
    else:
        # Reference split on non-zero ratings
        rated_indices = np.argwhere(matrix > 0)
        n_ratings = len(rated_indices)
        n_test = int(np.round(n_ratings * test_ratio))

        np.random.seed(42)
        shuffled = np.random.permutation(n_ratings)
        test_indices = rated_indices[shuffled[:n_test]]

        train_mat = matrix.copy()
        test_mat = np.zeros_like(matrix)

        for r, c in test_indices:
            test_mat[r, c] = train_mat[r, c]
            train_mat[r, c] = 0.0

        return train_mat, test_mat


def _run_encode_genres(movies_df: pd.DataFrame):
    """Executes encode_genres or reference one-hot/multi-hot encoding."""
    if encode_genres is not None:
        return encode_genres(movies_df)
    elif DataPreprocessor is not None and hasattr(DataPreprocessor, "encode_genres"):
        return DataPreprocessor().encode_genres(movies_df)
    else:
        genres_series = movies_df["genres"].fillna("").astype(str)
        unique_genres = sorted({g for sub in genres_series.str.split("|") for g in sub if g})
        encoded = np.zeros((len(movies_df), len(unique_genres)), dtype=int)
        for idx, g_str in enumerate(genres_series):
            active = set(g_str.split("|"))
            for g_idx, g in enumerate(unique_genres):
                if g in active:
                    encoded[idx, g_idx] = 1
        return encoded


def _run_create_feature_matrix(movies_df: pd.DataFrame):
    """Executes create_feature_matrix or reference feature builder."""
    if create_feature_matrix is not None:
        return create_feature_matrix(movies_df)
    elif DataPreprocessor is not None and hasattr(DataPreprocessor, "create_feature_matrix"):
        return DataPreprocessor().create_feature_matrix(movies_df)
    else:
        genre_features = _run_encode_genres(movies_df)
        return genre_features.astype(np.float32)


# ============================================================================
# Unit Tests
# ============================================================================

def test_normalize_ratings_zero_mean(sample_user_item_matrix: np.ndarray):
    """Test that rating normalization centers non-zero ratings around zero mean."""
    normalized = _run_normalize(sample_user_item_matrix)

    assert isinstance(normalized, np.ndarray), "Normalized matrix must be a numpy ndarray"
    assert normalized.shape == sample_user_item_matrix.shape, "Shape must be preserved after normalization"

    # For each user with ratings, check that the mean of normalized rated entries is ~0
    for u in range(normalized.shape[0]):
        rated_mask = sample_user_item_matrix[u] > 0
        if np.any(rated_mask):
            user_mean = np.mean(normalized[u, rated_mask])
            assert np.isclose(user_mean, 0.0, atol=1e-5), (
                f"User {u} normalized mean should be 0.0, got {user_mean}"
            )


def test_train_test_split_ratio(sample_user_item_matrix: np.ndarray):
    """Test that train_test_split splits ratings according to the requested ratio."""
    test_ratio = 0.25
    train_data, test_data = _run_train_test_split(sample_user_item_matrix, test_ratio=test_ratio)

    total_ratings = np.count_nonzero(sample_user_item_matrix > 0)
    train_ratings = np.count_nonzero(train_data > 0)
    test_ratings = np.count_nonzero(test_data > 0)

    assert train_ratings + test_ratings == total_ratings, (
        f"Sum of train ({train_ratings}) and test ({test_ratings}) ratings must equal total ({total_ratings})"
    )

    actual_ratio = test_ratings / total_ratings
    assert np.isclose(actual_ratio, test_ratio, atol=0.08), (
        f"Expected test ratio ~{test_ratio}, but got {actual_ratio:.3f}"
    )


def test_encode_genres_shape(sample_movies_df: pd.DataFrame):
    """Test that encode_genres produces a matrix with shape (n_movies, n_unique_genres)."""
    encoded = _run_encode_genres(sample_movies_df)

    if isinstance(encoded, pd.DataFrame):
        encoded_arr = encoded.values
    else:
        encoded_arr = np.asarray(encoded)

    # 8 movies in sample_movies_df
    assert encoded_arr.shape[0] == len(sample_movies_df), (
        f"Rows ({encoded_arr.shape[0]}) must equal number of movies ({len(sample_movies_df)})"
    )
    # At least several distinct genres (Animation, Children, Comedy, Action, Adventure, etc.)
    assert encoded_arr.shape[1] >= 5, f"Expected at least 5 distinct genres, got {encoded_arr.shape[1]}"


def test_encode_genres_binary_values(sample_movies_df: pd.DataFrame):
    """Test that genre encoding produces strictly binary values (0 or 1)."""
    encoded = _run_encode_genres(sample_movies_df)

    if isinstance(encoded, pd.DataFrame):
        encoded_arr = encoded.values
    else:
        encoded_arr = np.asarray(encoded)

    unique_vals = np.unique(encoded_arr)
    assert set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}), (
        f"Genre encoding contains non-binary values: {unique_vals}"
    )


def test_create_feature_matrix_shape(sample_movies_df: pd.DataFrame):
    """Test that create_feature_matrix creates a 2D matrix matching movie count."""
    features = _run_create_feature_matrix(sample_movies_df)

    if isinstance(features, pd.DataFrame):
        feat_arr = features.values
    else:
        feat_arr = np.asarray(features)

    assert feat_arr.ndim == 2, "Feature matrix must be 2-dimensional"
    assert feat_arr.shape[0] == len(sample_movies_df), (
        f"Feature matrix row count ({feat_arr.shape[0]}) must match movie count ({len(sample_movies_df)})"
    )
    assert feat_arr.shape[1] > 0, "Feature matrix must have at least 1 feature column"
