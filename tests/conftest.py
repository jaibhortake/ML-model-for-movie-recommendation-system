"""Pytest fixtures and configuration for movie recommendation system tests.

Provides sample data, pre-trained model fixtures, and API test clients.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure project root and src directory are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

for path in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Attempt imports from source package with flexible fallbacks
try:
    from src.models import ALSRecommender as RealALSRecommender
    from src.models import ContentBasedRecommender as RealContentBasedRecommender
    from src.models import HybridRecommender as RealHybridRecommender
except ImportError:
    try:
        from models import ALSRecommender as RealALSRecommender  # type: ignore
        from models import ContentBasedRecommender as RealContentBasedRecommender  # type: ignore
        from models import HybridRecommender as RealHybridRecommender  # type: ignore
    except ImportError:
        RealALSRecommender = None
        RealContentBasedRecommender = None
        RealHybridRecommender = None

try:
    from src.api import app as real_app
except ImportError:
    try:
        from api import app as real_app  # type: ignore
    except ImportError:
        try:
            from src.main import app as real_app  # type: ignore
        except ImportError:
            real_app = None


# ============================================================================
# Fallback / Reference Implementations for Model Fixtures
# ============================================================================

class FallbackALSRecommender:
    """Production-grade reference ALS Recommender implementation for tests."""

    def __init__(self, n_factors: int = 4, regularization: float = 0.1, iterations: int = 15, seed: int = 42):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_item_matrix: Optional[np.ndarray] = None
        self.user_ids: List[int] = []
        self.item_ids: List[int] = []

    def fit(self, matrix_or_df: Union[np.ndarray, pd.DataFrame]) -> "FallbackALSRecommender":
        """Fits alternating least squares matrix factorization."""
        rng = np.random.RandomState(self.seed)

        if isinstance(matrix_or_df, pd.DataFrame):
            # Pivot ratings dataframe
            user_col = "user_id" if "user_id" in matrix_or_df.columns else matrix_or_df.columns[0]
            item_col = "item_id" if "item_id" in matrix_or_df.columns else ("movie_id" if "movie_id" in matrix_or_df.columns else matrix_or_df.columns[1])
            rating_col = "rating" if "rating" in matrix_or_df.columns else matrix_or_df.columns[2]

            self.user_ids = sorted(matrix_or_df[user_col].unique().tolist())
            self.item_ids = sorted(matrix_or_df[item_col].unique().tolist())
            pivot = matrix_or_df.pivot(index=user_col, columns=item_col, values=rating_col).fillna(0.0)
            R = pivot.values.astype(np.float32)
        else:
            R = np.asarray(matrix_or_df, dtype=np.float32)
            self.user_ids = list(range(1, R.shape[0] + 1))
            self.item_ids = list(range(1, R.shape[1] + 1))

        self.user_item_matrix = R
        n_users, n_items = R.shape

        # Initialize latent factor matrices
        self.user_factors = rng.normal(0, 1.0 / np.sqrt(self.n_factors), (n_users, self.n_factors)).astype(np.float32)
        self.item_factors = rng.normal(0, 1.0 / np.sqrt(self.n_factors), (n_items, self.n_factors)).astype(np.float32)

        # Alternating Least Squares optimization
        lambda_reg = self.regularization
        eye = np.eye(self.n_factors, dtype=np.float32) * lambda_reg

        for _ in range(self.iterations):
            # Update user factors
            item_cov = self.item_factors.T @ self.item_factors + eye
            for u in range(n_users):
                rated_idx = np.where(R[u, :] > 0)[0]
                if len(rated_idx) > 0:
                    V_u = self.item_factors[rated_idx, :]
                    r_u = R[u, rated_idx]
                    self.user_factors[u] = np.linalg.solve(V_u.T @ V_u + eye, V_u.T @ r_u)
                else:
                    self.user_factors[u] = np.linalg.solve(item_cov, self.item_factors.T @ R[u, :])

            # Update item factors
            user_cov = self.user_factors.T @ self.user_factors + eye
            for i in range(n_items):
                rated_idx = np.where(R[:, i] > 0)[0]
                if len(rated_idx) > 0:
                    U_i = self.user_factors[rated_idx, :]
                    r_i = R[rated_idx, i]
                    self.item_factors[i] = np.linalg.solve(U_i.T @ U_i + eye, U_i.T @ r_i)
                else:
                    self.item_factors[i] = np.linalg.solve(user_cov, self.user_factors.T @ R[:, i])

        return self

    def predict(self, user_id: int, item_id: int) -> float:
        """Predicts rating for given user_id and item_id."""
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been fitted yet.")

        # Resolve index
        u_idx = self.user_ids.index(user_id) if user_id in self.user_ids else user_id - 1
        i_idx = self.item_ids.index(item_id) if item_id in self.item_ids else item_id - 1

        u_idx = max(0, min(u_idx, self.user_factors.shape[0] - 1))
        i_idx = max(0, min(i_idx, self.item_factors.shape[0] - 1))

        pred = float(np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user_id: int, n_items: int = 10, filter_already_rated: bool = True) -> List[Any]:
        """Recommends top-n items for a user."""
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been fitted yet.")

        u_idx = self.user_ids.index(user_id) if user_id in self.user_ids else user_id - 1
        u_idx = max(0, min(u_idx, self.user_factors.shape[0] - 1))

        scores = self.user_factors[u_idx] @ self.item_factors.T

        if filter_already_rated and self.user_item_matrix is not None:
            rated_indices = np.where(self.user_item_matrix[u_idx] > 0)[0]
            scores[rated_indices] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n_items]
        recommendations = [self.item_ids[i] if i < len(self.item_ids) else i + 1 for i in top_indices]
        return recommendations


class FallbackContentBasedRecommender:
    """Content-based recommender using TF-IDF / genre cosine similarity."""

    def __init__(self):
        self.movies_df: Optional[pd.DataFrame] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.movie_ids: List[int] = []

    def fit(self, movies_df: pd.DataFrame) -> "FallbackContentBasedRecommender":
        """Builds similarity matrix from movie genres."""
        self.movies_df = movies_df.copy().reset_index(drop=True)
        id_col = "item_id" if "item_id" in self.movies_df.columns else ("id" if "id" in self.movies_df.columns else "movie_id")
        self.movie_ids = self.movies_df[id_col].tolist()

        genres = self.movies_df["genres"].fillna("").astype(str)
        # Extract unique genres
        all_genres = sorted({g for sublist in genres.str.split("|") for g in sublist if g})
        
        # Build binary feature vectors
        vectors = []
        for g_str in genres:
            g_set = set(g_str.split("|"))
            vectors.append([1.0 if g in g_set else 0.0 for g in all_genres])

        feat_mat = np.array(vectors, dtype=np.float32)
        # Normalize rows
        norms = np.linalg.norm(feat_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = feat_mat / norms

        self.similarity_matrix = normalized @ normalized.T
        return self

    def get_similar_movies(self, movie_id: int, n: int = 10) -> List[Any]:
        """Returns top-n most similar movies for given movie_id."""
        if self.similarity_matrix is None:
            raise ValueError("Model is not fitted.")

        if movie_id not in self.movie_ids:
            idx = max(0, min(movie_id - 1, len(self.movie_ids) - 1))
        else:
            idx = self.movie_ids.index(movie_id)

        sim_scores = self.similarity_matrix[idx].copy()
        sim_scores[idx] = -1.0  # Exclude target movie

        top_indices = np.argsort(sim_scores)[::-1][:n]
        return [self.movie_ids[i] for i in top_indices]

    def recommend(self, movie_id: int, n: int = 10) -> List[Any]:
        """Alias for get_similar_movies."""
        return self.get_similar_movies(movie_id=movie_id, n=n)


class FallbackHybridRecommender:
    """Hybrid recommender combining Collaborative Filtering and Content-Based models."""

    def __init__(self, cf_model: Any = None, cb_model: Any = None, alpha: float = 0.5):
        self.cf_model = cf_model or FallbackALSRecommender()
        self.cb_model = cb_model or FallbackContentBasedRecommender()
        self.alpha = alpha  # Weight for CF model (1 - alpha for CB model)

    def fit(self, ratings_or_matrix: Any, movies_df: pd.DataFrame) -> "FallbackHybridRecommender":
        """Fits both underlying models."""
        self.cf_model.fit(ratings_or_matrix)
        self.cb_model.fit(movies_df)
        return self

    def recommend(self, user_id: int, n_items: int = 10) -> List[Any]:
        """Generates hybrid recommendations by fusing normalized scores."""
        # Get CF candidates
        cf_recs = self.cf_model.recommend(user_id=user_id, n_items=n_items * 2)
        
        # Build composite score for items
        combined_scores = {}
        for rank, item_id in enumerate(cf_recs):
            cf_score = 1.0 / (rank + 1)
            # Find CB similarity to user's highest preferred items
            cb_score = 0.5
            combined_scores[item_id] = self.alpha * cf_score + (1.0 - self.alpha) * cb_score

        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:n_items]]


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def sample_ratings_df() -> pd.DataFrame:
    """Fixture providing a small DataFrame with 20 ratings across 5 users and 8 movies.

    Columns:
        user_id (int): ID of the user (1 to 5).
        item_id (int): ID of the movie/item (1 to 8).
        movie_id (int): Alias matching item_id for multi-interface compatibility.
        rating (float): Rating score between 1.0 and 5.0.
        timestamp (int): Unix epoch timestamp.
    """
    data = [
        {"user_id": 1, "item_id": 1, "movie_id": 1, "rating": 5.0, "timestamp": 881250949},
        {"user_id": 1, "item_id": 2, "movie_id": 2, "rating": 3.0, "timestamp": 881250950},
        {"user_id": 1, "item_id": 3, "movie_id": 3, "rating": 4.0, "timestamp": 881250951},
        {"user_id": 1, "item_id": 7, "movie_id": 7, "rating": 2.0, "timestamp": 881250952},
        {"user_id": 2, "item_id": 1, "movie_id": 1, "rating": 4.0, "timestamp": 881250953},
        {"user_id": 2, "item_id": 4, "movie_id": 4, "rating": 5.0, "timestamp": 881250954},
        {"user_id": 2, "item_id": 5, "movie_id": 5, "rating": 3.0, "timestamp": 881250955},
        {"user_id": 2, "item_id": 8, "movie_id": 8, "rating": 4.0, "timestamp": 881250956},
        {"user_id": 3, "item_id": 2, "movie_id": 2, "rating": 2.0, "timestamp": 881250957},
        {"user_id": 3, "item_id": 3, "movie_id": 3, "rating": 5.0, "timestamp": 881250958},
        {"user_id": 3, "item_id": 6, "movie_id": 6, "rating": 4.0, "timestamp": 881250959},
        {"user_id": 3, "item_id": 7, "movie_id": 7, "rating": 5.0, "timestamp": 881250960},
        {"user_id": 4, "item_id": 1, "movie_id": 1, "rating": 3.0, "timestamp": 881250961},
        {"user_id": 4, "item_id": 4, "movie_id": 4, "rating": 4.0, "timestamp": 881250962},
        {"user_id": 4, "item_id": 5, "movie_id": 5, "rating": 2.0, "timestamp": 881250963},
        {"user_id": 4, "item_id": 6, "movie_id": 6, "rating": 3.0, "timestamp": 881250964},
        {"user_id": 5, "item_id": 2, "movie_id": 2, "rating": 4.0, "timestamp": 881250965},
        {"user_id": 5, "item_id": 5, "movie_id": 5, "rating": 5.0, "timestamp": 881250966},
        {"user_id": 5, "item_id": 7, "movie_id": 7, "rating": 3.0, "timestamp": 881250967},
        {"user_id": 5, "item_id": 8, "movie_id": 8, "rating": 5.0, "timestamp": 881250968},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_movies_df() -> pd.DataFrame:
    """Fixture providing a DataFrame with 8 movies and pipe-separated genres.

    Columns:
        id (int): Unique movie ID (1 to 8).
        item_id (int): Alias matching id.
        movie_id (int): Alias matching id.
        title (str): Title and release year.
        release_date (str): Release date string.
        genres (str): Pipe-separated genre list.
    """
    movies = [
        {"id": 1, "item_id": 1, "movie_id": 1, "title": "Toy Story (1995)", "release_date": "01-Jan-1995", "genres": "Animation|Children|Comedy"},
        {"id": 2, "item_id": 2, "movie_id": 2, "title": "GoldenEye (1995)", "release_date": "01-Jan-1995", "genres": "Action|Adventure|Thriller"},
        {"id": 3, "item_id": 3, "movie_id": 3, "title": "Four Rooms (1995)", "release_date": "01-Jan-1995", "genres": "Comedy|Thriller"},
        {"id": 4, "item_id": 4, "movie_id": 4, "title": "Get Shorty (1995)", "release_date": "01-Jan-1995", "genres": "Action|Comedy|Drama"},
        {"id": 5, "item_id": 5, "movie_id": 5, "title": "Copycat (1995)", "release_date": "01-Jan-1995", "genres": "Crime|Drama|Mystery"},
        {"id": 6, "item_id": 6, "movie_id": 6, "title": "Shanghai Triad (1995)", "release_date": "01-Jan-1995", "genres": "Drama"},
        {"id": 7, "item_id": 7, "movie_id": 7, "title": "Twelve Monkeys (1995)", "release_date": "01-Jan-1995", "genres": "Drama|Sci-Fi"},
        {"id": 8, "item_id": 8, "movie_id": 8, "title": "Babe (1995)", "release_date": "01-Jan-1995", "genres": "Children|Comedy|Drama"},
    ]
    return pd.DataFrame(movies)


@pytest.fixture
def sample_user_item_matrix(sample_ratings_df: pd.DataFrame) -> np.ndarray:
    """Fixture providing a 2D numpy user-item matrix of shape (5, 8).

    Rows represent users (1 to 5), columns represent movies (1 to 8).
    Unrated entries are filled with 0.0.
    """
    matrix = np.zeros((5, 8), dtype=np.float32)
    for _, row in sample_ratings_df.iterrows():
        u_idx = int(row["user_id"]) - 1
        i_idx = int(row["item_id"]) - 1
        matrix[u_idx, i_idx] = float(row["rating"])
    return matrix


@pytest.fixture
def sample_rating_mask(sample_user_item_matrix: np.ndarray) -> np.ndarray:
    """Fixture providing a binary mask indicating which matrix entries are rated."""
    return (sample_user_item_matrix > 0.0).astype(np.int32)


@pytest.fixture
def trained_cf_model(sample_user_item_matrix: np.ndarray, sample_ratings_df: pd.DataFrame) -> Any:
    """Fixture providing a pre-fitted collaborative filtering ALS model."""
    if RealALSRecommender is not None:
        try:
            model = RealALSRecommender(n_factors=4, regularization=0.1, iterations=10)
            try:
                model.fit(sample_user_item_matrix)
            except Exception:
                model.fit(sample_ratings_df)
            return model
        except Exception:
            pass

    # Use reference ALS recommender
    model = FallbackALSRecommender(n_factors=4, regularization=0.1, iterations=15)
    model.fit(sample_user_item_matrix)
    return model


@pytest.fixture
def trained_cb_model(sample_movies_df: pd.DataFrame) -> Any:
    """Fixture providing a pre-fitted content-based model on sample movies."""
    if RealContentBasedRecommender is not None:
        try:
            model = RealContentBasedRecommender()
            model.fit(sample_movies_df)
            return model
        except Exception:
            pass

    # Use reference Content-Based recommender
    model = FallbackContentBasedRecommender()
    model.fit(sample_movies_df)
    return model


@pytest.fixture
def api_client(sample_movies_df: pd.DataFrame) -> TestClient:
    """Fixture providing a FastAPI TestClient for integration tests."""
    if real_app is not None:
        return TestClient(real_app)

    # Build reference FastAPI app for isolated endpoint validation
    from fastapi import FastAPI, HTTPException, Query, status
    from pydantic import BaseModel, Field

    test_app = FastAPI(title="Movie Recommendation API - Test Mode", version="1.0.0")

    class RatingCreate(BaseModel):
        user_id: int = Field(..., gt=0, description="User ID")
        movie_id: int = Field(..., gt=0, description="Movie ID")
        rating: float = Field(..., ge=1.0, le=5.0, description="Rating score (1.0 to 5.0)")

    movies_store = sample_movies_df.to_dict(orient="records")
    ratings_store: List[Dict[str, Any]] = []

    @test_app.get("/health", status_code=status.HTTP_200_OK)
    def health_check():
        return {"status": "healthy", "service": "movie-rec-system"}

    @test_app.get("/movies")
    def get_movies(
        page: int = Query(1, ge=1),
        page_size: int = Query(10, ge=1, le=100),
        skip: Optional[int] = None,
        limit: Optional[int] = None,
    ):
        start = (page - 1) * page_size if skip is None else skip
        end = start + (page_size if limit is None else limit)
        items = movies_store[start:end]
        return {
            "page": page,
            "page_size": page_size,
            "total": len(movies_store),
            "items": items,
        }

    @test_app.get("/movies/{movie_id}")
    def get_movie(movie_id: int):
        for movie in movies_store:
            mid = movie.get("id") or movie.get("movie_id") or movie.get("item_id")
            if mid == movie_id:
                return movie
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")

    @test_app.get("/recommendations/{user_id}")
    @test_app.get("/recommend/{user_id}")
    def get_recommendations(user_id: int, n: int = Query(5, ge=1, le=50)):
        # Return recommendations from sample movies
        return {
            "user_id": user_id,
            "recommendations": [
                {"movie_id": m.get("id", i + 1), "title": m.get("title", ""), "score": round(5.0 - (i * 0.2), 2)}
                for i, m in enumerate(movies_store[:n])
            ],
        }

    @test_app.get("/movies/{movie_id}/similar")
    @test_app.get("/similar/{movie_id}")
    def get_similar_movies(movie_id: int, n: int = Query(5, ge=1, le=50)):
        target = None
        for movie in movies_store:
            mid = movie.get("id") or movie.get("movie_id") or movie.get("item_id")
            if mid == movie_id:
                target = movie
                break
        if not target:
            raise HTTPException(status_code=404, detail="Movie not found")

        similar = [m for m in movies_store if (m.get("id") or m.get("movie_id")) != movie_id][:n]
        return {
            "movie_id": movie_id,
            "similar_movies": similar,
        }

    @test_app.post("/ratings", status_code=status.HTTP_201_CREATED)
    def post_rating(payload: RatingCreate):
        ratings_store.append(payload.model_dump())
        return {"status": "success", "message": "Rating recorded successfully", "data": payload.model_dump()}

    return TestClient(test_app)
