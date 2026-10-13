"""Content-based recommendation model using TF-IDF and Cosine Similarity.

Constructs item representations from movie titles and genre annotations, computes
item-item pairwise similarities, and builds personalized user profile preference
vectors for recommendation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """Content-based recommender using TF-IDF features and cosine similarity."""

    def __init__(
        self,
        max_features: int = 5000,
        genre_weight: float = 1.5,
        min_df: int = 1,
    ) -> None:
        """Initialize hyperparameters.

        Args:
            max_features: Maximum number of TF-IDF vocabulary features.
            genre_weight: Multiplier weight applied to genre tokens relative to titles.
            min_df: Minimum document frequency threshold for n-grams.
        """
        self.max_features = max_features
        self.genre_weight = genre_weight
        self.min_df = min_df

        self.vectorizer: TfidfVectorizer | None = None
        self.item_features: csr_matrix | None = None  # L2-normalized feature matrix
        self.movies_df: pd.DataFrame | None = None
        self.movie_id_to_idx: dict[int, int] = {}
        self.idx_to_movie_id: dict[int, int] = {}
        self.movie_titles: dict[int, str] = {}
        self.movie_genres: dict[int, str] = {}

    def fit(self, movies_df: pd.DataFrame) -> ContentBasedRecommender:
        """Fit content feature representations on movies catalog.

        Args:
            movies_df: DataFrame containing at least `movie_id`, `title`, and
                either genre binary columns, `genre_list`, or `genres_str`.

        Returns:
            Fitted ContentBasedRecommender instance.
        """
        if "movie_id" not in movies_df.columns or "title" not in movies_df.columns:
            raise ValueError("movies_df must contain 'movie_id' and 'title' columns.")

        self.movies_df = movies_df.copy().reset_index(drop=True)
        n_items = len(self.movies_df)

        self.movie_id_to_idx = {
            int(mid): i for i, mid in enumerate(self.movies_df["movie_id"])
        }
        self.idx_to_movie_id = {
            i: int(mid) for mid, i in self.movie_id_to_idx.items()
        }

        # Build composite text representation: cleaned title + repeated genres
        corpus: list[str] = []
        for _, row in self.movies_df.iterrows():
            mid = int(row["movie_id"])
            title = str(row["title"])
            # Remove year in parentheses e.g. "Toy Story (1995)" -> "Toy Story"
            clean_title = pd.Series([title]).str.replace(r"\s*\(\d{4}\)$", "", regex=True).iloc[0].lower()
            self.movie_titles[mid] = title

            # Extract genres
            genre_tokens: list[str] = []
            if "genre_list" in row and isinstance(row["genre_list"], (list, tuple)):
                genre_tokens = [str(g).lower().replace("-", "").replace("'", "") for g in row["genre_list"]]
            elif "genres_str" in row and pd.notna(row["genres_str"]):
                genre_tokens = [str(g).lower().replace("-", "").replace("'", "") for g in str(row["genres_str"]).split("|")]
            elif "genres" in row and pd.notna(row["genres"]):
                genre_tokens = [str(g).lower().replace("-", "").replace("'", "") for g in str(row["genres"]).split("|")]
            else:
                # Look for binary genre flags
                for col in self.movies_df.columns:
                    if col not in {"movie_id", "title", "release_date", "video_release_date", "imdb_url"} and row.get(col, 0) == 1:
                        genre_tokens.append(str(col).lower().replace("-", "").replace("'", ""))

            self.movie_genres[mid] = " ".join(genre_tokens)

            # Weight genre tokens by repeating them according to genre_weight
            repeated_genres = " ".join(genre_tokens * int(round(self.genre_weight)))
            content_text = f"{clean_title} {repeated_genres}"
            corpus.append(content_text)

        logger.info(
            "Fitting TF-IDF vectorizer across %d items (max_features=%d)...",
            n_items,
            self.max_features,
        )
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=self.min_df,
            sublinear_tf=True,
        )
        raw_tfidf = self.vectorizer.fit_transform(corpus)
        # L2-normalize so cosine similarity is a fast dot product
        self.item_features = normalize(raw_tfidf, norm="l2", axis=1)

        logger.info(
            "Content-based item features successfully constructed: shape %s",
            self.item_features.shape,
        )
        return self

    def get_similar_movies(
        self,
        movie_id: int,
        n: int = 10,
    ) -> list[tuple[int, float]]:
        """Find the top-N most similar movies using cosine similarity.

        Args:
            movie_id: Target movie identifier.
            n: Number of similar movies to retrieve.

        Returns:
            List of (movie_id, similarity_score) sorted in descending order.
        """
        if self.item_features is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if movie_id not in self.movie_id_to_idx:
            logger.warning("Movie ID %d not found in catalog.", movie_id)
            return []

        idx = self.movie_id_to_idx[movie_id]
        target_vec = self.item_features.getrow(idx)

        # Dot product with all items (since vectors are L2-normalized, dot product == cosine similarity)
        sim_scores = (self.item_features @ target_vec.T).toarray().flatten()

        # Exclude self
        sim_scores[idx] = -1.0

        top_indices = np.argsort(sim_scores)[::-1][:n]
        results: list[tuple[int, float]] = []
        for i in top_indices:
            score = float(sim_scores[i])
            if score <= 0:
                break
            results.append((self.idx_to_movie_id[i], score))

        return results

    def recommend_for_user(
        self,
        user_id: int,
        user_ratings: pd.DataFrame,
        n: int = 10,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating",
        exclude_rated: bool = True,
    ) -> list[tuple[int, float]]:
        """Generate content-based recommendations for a user based on their rated history.

        Constructs a weighted user preference profile vector:
            u_vec = sum_i ((rating_i - mean_rating) * item_vec_i)
        and ranks candidate items by cosine similarity to this profile vector.

        Args:
            user_id: Target user ID.
            user_ratings: DataFrame containing user's ratings.
            n: Number of recommendations.
            user_col: User ID column.
            item_col: Item ID column.
            rating_col: Rating column.
            exclude_rated: Exclude already watched movies.

        Returns:
            List of (movie_id, predicted_score) tuples sorted in descending order.
        """
        if self.item_features is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        # Filter ratings for this user if user_col is present
        if user_col in user_ratings.columns:
            u_df = user_ratings[user_ratings[user_col] == user_id]
        else:
            u_df = user_ratings

        if u_df.empty:
            logger.warning("No rating history available for user %d in content recommender.", user_id)
            return []

        # Find items that exist in our feature catalog
        rated_items = u_df[item_col].tolist()
        ratings = u_df[rating_col].to_numpy(dtype=np.float32)

        valid_indices: list[int] = []
        valid_weights: list[float] = []
        rated_set: set[int] = set()

        mean_rating = float(np.mean(ratings)) if len(ratings) > 0 else 3.5

        for item, rating in zip(rated_items, ratings):
            item_int = int(item)
            rated_set.add(item_int)
            if item_int in self.movie_id_to_idx:
                idx = self.movie_id_to_idx[item_int]
                valid_indices.append(idx)
                # Mean-centered rating weight (favor movies rated above user average)
                valid_weights.append(float(rating) - mean_rating)

        if not valid_indices:
            logger.warning("None of the user's rated items exist in content feature space.")
            return []

        # Build user profile vector
        weights_arr = np.array(valid_weights, dtype=np.float32)
        # If all weights are 0 (e.g. all ratings identical), use uncentered ratings
        if np.all(weights_arr == 0):
            weights_arr = np.array(ratings, dtype=np.float32)

        subset_features = self.item_features[valid_indices]
        # Weighted combination: (n_rated, n_features).T @ weights -> (n_features, 1)
        profile_vec = (subset_features.T @ weights_arr).reshape(1, -1)
        profile_norm = np.linalg.norm(profile_vec)
        if profile_norm > 0:
            profile_vec = profile_vec / profile_norm

        # Compute cosine similarity between user profile and all movies
        scores = (self.item_features @ profile_vec.T).flatten()

        if exclude_rated:
            for item in rated_set:
                if item in self.movie_id_to_idx:
                    scores[self.movie_id_to_idx[item]] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n]
        recommendations: list[tuple[int, float]] = []
        for i in top_indices:
            score = float(scores[i])
            if score == -np.inf:
                break
            # Rescale cosine similarity [-1, 1] into estimated rating [1.0, 5.0]
            est_rating = float(np.clip(mean_rating + score * 1.5, 1.0, 5.0))
            recommendations.append((self.idx_to_movie_id[i], est_rating))

        return recommendations

    def save(self, filepath: str | Path) -> None:
        """Save fitted content model to disk using joblib.

        Args:
            filepath: Destination file path.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved ContentBasedRecommender to %s", path)

    @classmethod
    def load(cls, filepath: str | Path) -> ContentBasedRecommender:
        """Load model from disk.

        Args:
            filepath: Source file path.

        Returns:
            Restored ContentBasedRecommender instance.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Content model not found at {path}")
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object {type(model)} is not ContentBasedRecommender")
        logger.info("Loaded ContentBasedRecommender from %s", path)
        return model
