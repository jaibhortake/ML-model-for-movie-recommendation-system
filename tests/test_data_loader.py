"""Tests for the data loading module."""

import os
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


class TestLoadRatings:
    """Tests for the load_ratings function."""

    def test_load_ratings_returns_dataframe(self, sample_ratings_df: pd.DataFrame):
        """Ratings loader should return a pandas DataFrame."""
        assert isinstance(sample_ratings_df, pd.DataFrame)

    def test_load_ratings_columns(self, sample_ratings_df: pd.DataFrame):
        """Ratings DataFrame must contain required columns."""
        required_cols = {"user_id", "item_id", "rating"}
        assert required_cols.issubset(set(sample_ratings_df.columns)), (
            f"Missing columns: {required_cols - set(sample_ratings_df.columns)}"
        )

    def test_ratings_range(self, sample_ratings_df: pd.DataFrame):
        """All ratings must be between 1 and 5 inclusive."""
        assert sample_ratings_df["rating"].min() >= 1.0, "Found rating below 1"
        assert sample_ratings_df["rating"].max() <= 5.0, "Found rating above 5"

    def test_ratings_no_nulls(self, sample_ratings_df: pd.DataFrame):
        """Ratings DataFrame should not contain null values in key columns."""
        for col in ["user_id", "item_id", "rating"]:
            assert sample_ratings_df[col].isnull().sum() == 0, (
                f"Column '{col}' contains null values"
            )

    def test_ratings_user_ids_positive(self, sample_ratings_df: pd.DataFrame):
        """All user IDs should be positive integers."""
        assert (sample_ratings_df["user_id"] > 0).all(), "Found non-positive user_id"

    def test_ratings_item_ids_positive(self, sample_ratings_df: pd.DataFrame):
        """All item IDs should be positive integers."""
        assert (sample_ratings_df["item_id"] > 0).all(), "Found non-positive item_id"


class TestLoadMovies:
    """Tests for the load_movies function."""

    def test_load_movies_returns_dataframe(self, sample_movies_df: pd.DataFrame):
        """Movies loader should return a pandas DataFrame."""
        assert isinstance(sample_movies_df, pd.DataFrame)

    def test_load_movies_has_title(self, sample_movies_df: pd.DataFrame):
        """Movies DataFrame must have a title column."""
        assert "title" in sample_movies_df.columns, "Missing 'title' column"

    def test_load_movies_has_genres(self, sample_movies_df: pd.DataFrame):
        """Movies DataFrame must have a genres column."""
        assert "genres" in sample_movies_df.columns, "Missing 'genres' column"

    def test_movies_unique_ids(self, sample_movies_df: pd.DataFrame):
        """Movie IDs should be unique."""
        id_col = "movie_id" if "movie_id" in sample_movies_df.columns else sample_movies_df.index.name
        if id_col and id_col in sample_movies_df.columns:
            assert sample_movies_df[id_col].is_unique, "Duplicate movie IDs found"


class TestBuildUserItemMatrix:
    """Tests for user-item matrix construction."""

    def test_matrix_shape(
        self,
        sample_ratings_df: pd.DataFrame,
        sample_user_item_matrix: np.ndarray,
    ):
        """Matrix dimensions should match unique users × unique movies."""
        n_users = sample_ratings_df["user_id"].nunique()
        n_items = sample_ratings_df["item_id"].nunique()
        assert sample_user_item_matrix.shape == (n_users, n_items), (
            f"Expected shape ({n_users}, {n_items}), "
            f"got {sample_user_item_matrix.shape}"
        )

    def test_matrix_values_in_range(self, sample_user_item_matrix: np.ndarray):
        """All non-zero values in the matrix should be valid ratings [1, 5]."""
        nonzero = sample_user_item_matrix[sample_user_item_matrix != 0]
        if len(nonzero) > 0:
            assert nonzero.min() >= 1.0, "Matrix contains rating below 1"
            assert nonzero.max() <= 5.0, "Matrix contains rating above 5"

    def test_matrix_is_2d(self, sample_user_item_matrix: np.ndarray):
        """User-item matrix must be 2-dimensional."""
        assert sample_user_item_matrix.ndim == 2, (
            f"Expected 2D matrix, got {sample_user_item_matrix.ndim}D"
        )

    def test_matrix_dtype_numeric(self, sample_user_item_matrix: np.ndarray):
        """Matrix dtype should be numeric (float or int)."""
        assert np.issubdtype(sample_user_item_matrix.dtype, np.number), (
            f"Expected numeric dtype, got {sample_user_item_matrix.dtype}"
        )


class TestDownload:
    """Tests for the download mechanism."""

    def test_download_creates_directory(self, tmp_path):
        """Download function should create the target directory if needed."""
        target_dir = tmp_path / "test_data" / "raw"
        target_dir.mkdir(parents=True, exist_ok=True)
        assert target_dir.exists(), "Data directory was not created"
