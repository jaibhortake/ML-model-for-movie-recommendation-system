"""Hybrid Recommendation System (Collaborative + Content-Based).

Combines Matrix Factorization ALS (collaborative filtering) and TF-IDF
Content-Based filtering using Netflix-style adaptive weighting and graceful
cold-start fallback with transparent recommendation explanations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.preprocessor import build_sparse_matrix
from src.models.collaborative import MatrixFactorizationALS
from src.models.content_based import ContentBasedRecommender

logger = logging.getLogger(__name__)


class HybridRecommender:
    """Netflix-style Hybrid Recommender combining Collaborative and Content-based models."""

    def __init__(
        self,
        cf_weight: float = 0.7,
        cb_weight: float = 0.3,
        cold_start_threshold: int = 5,
        n_factors: int = 50,
        n_iterations: int = 20,
        reg: float = 0.1,
    ) -> None:
        """Initialize hybrid parameters and sub-models.

        Args:
            cf_weight: Default weight for collaborative filtering (alpha).
            cb_weight: Default weight for content-based filtering (1 - alpha).
            cold_start_threshold: Minimum number of user ratings below which
                the recommender shifts primary weight to content-based.
            n_factors: Latent factors for ALS CF model.
            n_iterations: Iteration steps for ALS CF model.
            reg: Regularization parameter for ALS CF model.
        """
        total_weight = cf_weight + cb_weight
        self.cf_weight = cf_weight / total_weight
        self.cb_weight = cb_weight / total_weight
        self.cold_start_threshold = cold_start_threshold

        self.cf_model = MatrixFactorizationALS(
            n_factors=n_factors,
            n_iterations=n_iterations,
            reg=reg,
        )
        self.cb_model = ContentBasedRecommender()

        self.ratings_df: pd.DataFrame | None = None
        self.movies_df: pd.DataFrame | None = None
        self.user_rating_counts: dict[int, int] = {}
        self.movie_titles: dict[int, str] = {}
        self.movie_genres: dict[int, str] = {}

    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating",
    ) -> HybridRecommender:
        """Fit both Collaborative and Content-Based sub-models.

        Args:
            ratings_df: DataFrame with user interactions.
            movies_df: DataFrame with movie metadata.
            user_col: Column name for user ID.
            item_col: Column name for item ID.
            rating_col: Column name for rating.

        Returns:
            Fitted HybridRecommender instance.
        """
        logger.info(
            "Fitting HybridRecommender with %d ratings and %d movies...",
            len(ratings_df),
            len(movies_df),
        )
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()

        # Cache titles and genres
        for _, row in self.movies_df.iterrows():
            mid = int(row["movie_id"])
            self.movie_titles[mid] = str(row["title"])
            if "genres_str" in row and pd.notna(row["genres_str"]):
                self.movie_genres[mid] = str(row["genres_str"])
            elif "genres" in row and pd.notna(row["genres"]):
                self.movie_genres[mid] = str(row["genres"])
            else:
                self.movie_genres[mid] = "Unknown"

        # Record user activity levels for adaptive cold-start weighting
        self.user_rating_counts = ratings_df[user_col].value_counts().to_dict()

        # 1. Fit Collaborative Filtering model
        logger.info("Step 1/2: Fitting Collaborative Filtering (ALS)...")
        sparse_mat, u2idx, idx2u, i2idx, idx2i = build_sparse_matrix(
            ratings_df, user_col=user_col, item_col=item_col, rating_col=rating_col
        )
        self.cf_model.fit(sparse_mat, user_to_idx=u2idx, item_to_idx=i2idx)

        # 2. Fit Content-Based model
        logger.info("Step 2/2: Fitting Content-Based Recommender (TF-IDF)...")
        self.cb_model.fit(self.movies_df)

        logger.info("HybridRecommender training successfully completed.")
        return self

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating",
    ) -> list[dict[str, Any]]:
        """Generate top-N hybrid recommendations with adaptive cold-start weights.

        Args:
            user_id: Target user ID.
            n: Number of recommendations to generate.
            exclude_rated: Whether to filter out items already rated by the user.
            user_col: User ID column name.
            item_col: Item ID column name.
            rating_col: Rating column name.

        Returns:
            List of recommendation dicts containing:
                - `movie_id`: Recommended movie ID.
                - `title`: Movie title.
                - `genres`: Genres description.
                - `score`: Final blended recommendation score.
                - `cf_score`: Collaborative filtering score.
                - `cb_score`: Content-based score.
                - `strategy`: Selection regime ('blended', 'content_cold_start', etc.).
        """
        if self.ratings_df is None or self.movies_df is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        # Determine user rating count
        user_count = self.user_rating_counts.get(user_id, 0)
        is_cold_start_user = user_count < self.cold_start_threshold

        # User rated items set
        user_ratings_subset = self.ratings_df[self.ratings_df[user_col] == user_id]
        rated_item_ids = set(user_ratings_subset[item_col].tolist()) if not user_ratings_subset.empty else set()

        # Dynamic weighting based on user interaction history
        if is_cold_start_user:
            effective_cf_w = 0.15
            effective_cb_w = 0.85
            strategy = "content_cold_start"
        else:
            effective_cf_w = self.cf_weight
            effective_cb_w = self.cb_weight
            strategy = "blended"

        # Generate candidate pool from both CF and CB
        candidate_item_ids: set[int] = set()
        cf_recs = self.cf_model.recommend(user_id, n=n * 3, exclude_rated=exclude_rated)
        for mid, _ in cf_recs:
            candidate_item_ids.add(mid)

        cb_recs = self.cb_model.recommend_for_user(
            user_id=user_id,
            user_ratings=user_ratings_subset,
            n=n * 3,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
            exclude_rated=exclude_rated,
        )
        for mid, _ in cb_recs:
            candidate_item_ids.add(mid)

        if not candidate_item_ids:
            # Fallback to popular items if neither returned candidates
            top_movies = self.movies_df["movie_id"].head(n).tolist()
            candidate_item_ids.update(top_movies)

        cf_dict = dict(cf_recs)
        cb_dict = dict(cb_recs)

        scored_candidates: list[dict[str, Any]] = []
        for mid in candidate_item_ids:
            if exclude_rated and mid in rated_item_ids:
                continue

            cf_s = cf_dict.get(mid)
            if cf_s is None:
                cf_s = self.cf_model.predict(user_id, mid)

            cb_s = cb_dict.get(mid)
            if cb_s is None:
                cb_s = self.cf_model.global_mean

            final_score = float(effective_cf_w * cf_s + effective_cb_w * cb_s)
            scored_candidates.append(
                {
                    "movie_id": mid,
                    "title": self.movie_titles.get(mid, f"Movie {mid}"),
                    "genres": self.movie_genres.get(mid, "Unknown"),
                    "score": round(final_score, 4),
                    "cf_score": round(cf_s, 4),
                    "cb_score": round(cb_s, 4),
                    "strategy": strategy,
                }
            )

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates[:n]

    def get_explanation(
        self,
        user_id: int,
        movie_id: int,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating",
    ) -> dict[str, Any]:
        """Provide transparent Netflix-style explanation for a movie recommendation.

        Explains why a movie was recommended:
        - Quantifies collaborative interaction strength
        - Identifies top previously rated movies that triggered content similarity
        - Outlines shared genres and weights

        Args:
            user_id: Target user ID.
            movie_id: Target movie ID.
            user_col: User ID column name.
            item_col: Item ID column name.
            rating_col: Rating column name.

        Returns:
            Dictionary explaining recommendation drivers and similar items.
        """
        if self.ratings_df is None or self.movies_df is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        cf_pred = self.cf_model.predict(user_id, movie_id)
        user_count = self.user_rating_counts.get(user_id, 0)
        target_title = self.movie_titles.get(movie_id, f"Movie {movie_id}")
        target_genres = self.movie_genres.get(movie_id, "Unknown")

        # Find user's top-rated movies to locate the content-based anchor
        user_history = self.ratings_df[self.ratings_df[user_col] == user_id]
        top_user_movies = (
            user_history.sort_values(by=rating_col, ascending=False)
            .head(5)[item_col]
            .tolist()
            if not user_history.empty
            else []
        )

        similar_liked_items: list[dict[str, Any]] = []
        for prev_mid in top_user_movies:
            prev_mid = int(prev_mid)
            if prev_mid == movie_id:
                continue
            sim_list = self.cb_model.get_similar_movies(prev_mid, n=50)
            sim_map = dict(sim_list)
            if movie_id in sim_map:
                similar_liked_items.append(
                    {
                        "movie_id": prev_mid,
                        "title": self.movie_titles.get(prev_mid, f"Movie {prev_mid}"),
                        "similarity": round(sim_map[movie_id], 3),
                    }
                )

        similar_liked_items.sort(key=lambda x: x["similarity"], reverse=True)

        is_cold = user_count < self.cold_start_threshold
        if is_cold:
            reason = (
                f"Recommended based on content features matching your early preferences "
                f"(genre: {target_genres})."
            )
        elif similar_liked_items:
            best_match = similar_liked_items[0]["title"]
            reason = f"Recommended because you liked '{best_match}' and related {target_genres} films."
        else:
            reason = f"Recommended based on viewing trends from users with taste profiles similar to yours."

        return {
            "user_id": user_id,
            "movie_id": movie_id,
            "title": target_title,
            "genres": target_genres,
            "predicted_cf_rating": round(cf_pred, 3),
            "user_rating_count": user_count,
            "cold_start_mode": is_cold,
            "similar_liked_movies": similar_liked_items[:3],
            "reason": reason,
        }

    def save(self, filepath: str | Path) -> None:
        """Save hybrid recommender state to disk.

        Args:
            filepath: Destination file path.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved HybridRecommender to %s", path)

    @classmethod
    def load(cls, filepath: str | Path) -> HybridRecommender:
        """Load hybrid recommender from disk.

        Args:
            filepath: Source file path.

        Returns:
            Restored HybridRecommender instance.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Hybrid model not found at {path}")
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object {type(model)} is not HybridRecommender")
        logger.info("Loaded HybridRecommender from %s", path)
        return model
