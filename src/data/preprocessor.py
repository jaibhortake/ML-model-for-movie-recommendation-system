"""Data preprocessing and feature engineering module.

Provides rating normalization, chronological temporal splitting, sparse matrix
construction, multi-hot genre encoding, and text TF-IDF vectorization.
"""

from __future__ import annotations

import logging
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)

DEFAULT_GENRE_NAMES: Final[list[str]] = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def normalize_ratings(
    ratings_df: pd.DataFrame,
    user_col: str = "user_id",
    rating_col: str = "rating",
    min_ratings: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Normalize ratings using per-user mean centering.

    Mean-centering removes user bias (some users systematically rate higher or
    lower than average) while preserving relative preferences.

    Args:
        ratings_df: DataFrame containing user ratings.
        user_col: Column name representing user identifiers.
        rating_col: Column name representing numeric ratings.
        min_ratings: Minimum number of ratings required for a user (users with
            fewer ratings use the global mean).

    Returns:
        Tuple containing:
            - Copy of DataFrame with new column `normalized_rating` and `user_mean`.
            - Series mapping each user_id to their mean rating.
    """
    if rating_col not in ratings_df.columns or user_col not in ratings_df.columns:
        raise KeyError(f"Expected '{user_col}' and '{rating_col}' in DataFrame columns.")

    df = ratings_df.copy()
    global_mean = float(df[rating_col].mean())

    user_stats = df.groupby(user_col)[rating_col].agg(["mean", "count"])
    user_stats["smoothed_mean"] = np.where(
        user_stats["count"] >= min_ratings,
        user_stats["mean"],
        global_mean,
    )
    user_means = user_stats["smoothed_mean"]

    df["user_mean"] = df[user_col].map(user_means).fillna(global_mean).astype(np.float32)
    df["normalized_rating"] = (df[rating_col] - df["user_mean"]).astype(np.float32)

    logger.info(
        "Normalized %d ratings across %d users. Global mean: %.3f, mean range: [%.2f, %.2f]",
        len(df),
        len(user_means),
        global_mean,
        float(user_means.min()),
        float(user_means.max()),
    )
    return df, user_means


def train_test_split(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2,
    temporal: bool = True,
    timestamp_col: str = "timestamp",
    user_col: str = "user_id",
    per_user_temporal: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into train and test sets using temporal or random strategy.

    In production recommendation systems, evaluating on temporal holdouts
    accurately simulates future prediction scenarios and prevents data leakage.

    Args:
        ratings_df: Input ratings DataFrame.
        test_ratio: Proportion of ratings to hold out for evaluation (default: 0.2).
        temporal: If True, uses temporal splitting; if False, uses random split.
        timestamp_col: Column containing timestamp or datetime values.
        user_col: User ID column for stratified per-user temporal splits.
        per_user_temporal: If True, hold out the most recent test_ratio ratings
            for each individual user. If False, hold out globally latest ratings.
        random_state: Seed for reproducible random splitting if temporal=False.

    Returns:
        Tuple of (train_df, test_df).
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")

    if not temporal:
        logger.info("Performing random train/test split with test_ratio=%.2f", test_ratio)
        shuffled = ratings_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        split_idx = int(len(shuffled) * (1.0 - test_ratio))
        train_df = shuffled.iloc[:split_idx].copy()
        test_df = shuffled.iloc[split_idx:].copy()
        return train_df, test_df

    if timestamp_col not in ratings_df.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' required for temporal split.")

    df_sorted = ratings_df.sort_values(by=timestamp_col).reset_index(drop=True)

    if per_user_temporal and user_col in ratings_df.columns:
        logger.info("Performing per-user temporal train/test split (test_ratio=%.2f)", test_ratio)
        train_indices: list[int] = []
        test_indices: list[int] = []

        grouped = df_sorted.groupby(user_col, sort=False)
        for _, group in grouped:
            n_ratings = len(group)
            if n_ratings <= 1:
                train_indices.extend(group.index.tolist())
            else:
                n_test = max(1, int(round(n_ratings * test_ratio)))
                n_train = n_ratings - n_test
                train_indices.extend(group.index[:n_train].tolist())
                test_indices.extend(group.index[n_train:].tolist())

        train_df = df_sorted.loc[train_indices].sort_values(by=timestamp_col).reset_index(drop=True)
        test_df = df_sorted.loc[test_indices].sort_values(by=timestamp_col).reset_index(drop=True)
    else:
        logger.info("Performing global temporal train/test split (test_ratio=%.2f)", test_ratio)
        split_idx = int(len(df_sorted) * (1.0 - test_ratio))
        train_df = df_sorted.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = df_sorted.iloc[split_idx:].copy().reset_index(drop=True)

    logger.info(
        "Temporal split complete: %d train records (%.1f%%), %d test records (%.1f%%)",
        len(train_df),
        100.0 * len(train_df) / len(ratings_df),
        len(test_df),
        100.0 * len(test_df) / len(ratings_df),
    )
    return train_df, test_df


def build_sparse_matrix(
    ratings_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    user_to_idx: dict[int, int] | None = None,
    item_to_idx: dict[int, int] | None = None,
) -> tuple[csr_matrix, dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    """Build a SciPy CSR sparse matrix along with bidirectional ID index mappings.

    Args:
        ratings_df: DataFrame containing user interactions.
        user_col: User ID column name.
        item_col: Item ID column name.
        rating_col: Rating / interaction weight column name.
        user_to_idx: Optional existing mapping for users (for test set alignment).
        item_to_idx: Optional existing mapping for items (for test set alignment).

    Returns:
        Tuple containing:
            - csr_matrix of shape (n_users, n_items).
            - user_to_idx mapping (user_id -> row index).
            - idx_to_user mapping (row index -> user_id).
            - item_to_idx mapping (item_id -> col index).
            - idx_to_item mapping (col index -> item_id).
    """
    if user_to_idx is None:
        unique_users = sorted(ratings_df[user_col].unique())
        user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    idx_to_user = {i: uid for uid, i in user_to_idx.items()}

    if item_to_idx is None:
        unique_items = sorted(ratings_df[item_col].unique())
        item_to_idx = {iid: j for j, iid in enumerate(unique_items)}
    idx_to_item = {j: iid for iid, j in item_to_idx.items()}

    # Filter out entries not in the vocabulary mappings
    valid_mask = ratings_df[user_col].isin(user_to_idx) & ratings_df[item_col].isin(item_to_idx)
    valid_df = ratings_df[valid_mask]

    rows = valid_df[user_col].map(user_to_idx).to_numpy(dtype=np.int32)
    cols = valid_df[item_col].map(item_to_idx).to_numpy(dtype=np.int32)
    vals = valid_df[rating_col].to_numpy(dtype=np.float32)

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    mat = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    logger.info(
        "Constructed CSR sparse matrix shape (%d, %d) with %d non-zero entries (density: %.4f%%)",
        n_users,
        n_items,
        mat.nnz,
        100.0 * mat.nnz / (n_users * n_items) if (n_users * n_items) > 0 else 0.0,
    )
    return mat, user_to_idx, idx_to_user, item_to_idx, idx_to_item


def encode_genres(
    movies_df: pd.DataFrame,
    genre_cols: list[str] | None = None,
) -> np.ndarray:
    """Encode movie genres into multi-hot binary vectors.

    Args:
        movies_df: DataFrame containing movie metadata.
        genre_cols: List of binary genre column names. If None, checks for standard
            MovieLens genre columns, or uses `genre_list` / `genres_str`.

    Returns:
        2D numpy array of shape (n_movies, n_genres) with multi-hot binary encodings.
    """
    if genre_cols is None:
        matched_cols = [c for c in DEFAULT_GENRE_NAMES if c in movies_df.columns]
        if matched_cols:
            logger.info("Found %d pre-existing genre flag columns in movies_df.", len(matched_cols))
            return movies_df[matched_cols].to_numpy(dtype=np.float32)

    if "genre_list" in movies_df.columns:
        logger.info("Encoding genres from 'genre_list' column using MultiLabelBinarizer.")
        mlb = MultiLabelBinarizer()
        encoded = mlb.fit_transform(movies_df["genre_list"]).astype(np.float32)
        return encoded

    if "genres" in movies_df.columns or "genres_str" in movies_df.columns:
        col = "genres" if "genres" in movies_df.columns else "genres_str"
        logger.info("Encoding genres from '%s' delimited string using MultiLabelBinarizer.", col)
        genres_series = movies_df[col].fillna("").astype(str).str.split("|")
        mlb = MultiLabelBinarizer()
        encoded = mlb.fit_transform(genres_series).astype(np.float32)
        return encoded

    raise ValueError("movies_df does not contain recognizable genre columns or lists.")


def create_feature_matrix(
    movies_df: pd.DataFrame,
    title_col: str = "title",
    genre_cols: list[str] | None = None,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    include_genres: bool = True,
    genre_weight: float = 1.5,
) -> tuple[csr_matrix, TfidfVectorizer, np.ndarray]:
    """Create a combined TF-IDF feature matrix from movie titles and genres.

    Args:
        movies_df: DataFrame containing movie records.
        title_col: Column name containing movie titles.
        genre_cols: Optional list of binary genre columns.
        max_features: Maximum number of TF-IDF vocabulary features.
        ngram_range: Lower and upper boundary of range of n-values for n-grams.
        include_genres: Whether to concatenate weighted genre multi-hot vectors.
        genre_weight: Multiplier weight applied to genre features relative to title TF-IDF.

    Returns:
        Tuple containing:
            - Combined feature matrix as scipy csr_matrix (n_movies, n_features).
            - Fitted TfidfVectorizer instance.
            - Array of movie IDs corresponding to rows of the feature matrix.
    """
    if title_col not in movies_df.columns:
        raise KeyError(f"Column '{title_col}' not found in movies_df.")

    # Clean movie titles (strip year in parentheses for cleaner textual matching, e.g., 'Toy Story (1995)' -> 'Toy Story')
    cleaned_titles = (
        movies_df[title_col]
        .fillna("")
        .astype(str)
        .str.replace(r"\s*\(\d{4}\)$", "", regex=True)
        .str.lower()
    )

    logger.info("Fitting TfidfVectorizer on %d titles (max_features=%d)...", len(cleaned_titles), max_features)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    title_tfidf = vectorizer.fit_transform(cleaned_titles)

    movie_ids = movies_df["movie_id"].to_numpy(dtype=np.int32) if "movie_id" in movies_df.columns else np.arange(len(movies_df))

    if not include_genres:
        return title_tfidf, vectorizer, movie_ids

    genres_encoded = encode_genres(movies_df, genre_cols=genre_cols) * float(genre_weight)
    genres_sparse = csr_matrix(genres_encoded)

    from scipy.sparse import hstack

    combined_matrix = hstack([title_tfidf, genres_sparse], format="csr")
    logger.info(
        "Constructed combined feature matrix of shape (%d, %d) with title TF-IDF + weighted genres.",
        combined_matrix.shape[0],
        combined_matrix.shape[1],
    )
    return combined_matrix, vectorizer, movie_ids
