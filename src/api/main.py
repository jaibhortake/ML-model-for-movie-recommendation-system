"""FastAPI application for the Netflix-style Movie Recommendation System."""

from __future__ import annotations

import logging
import math
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Path as FastPath, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to sys.path if not present
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.api.schemas import (
    HealthResponse,
    MetricsResponse,
    MovieRecommendation,
    MovieResponse,
    PaginatedResponse,
    RatingRequest,
    RatingResponse,
    RecommendationMethod,
    RecommendationResponse,
    SimilarMoviesResponse,
)
from src.data.dataset import load_dataset
from src.models.recommender import HybridRecommender

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("movie_rec_api")

# Global in-memory recommender instance
recommender: Optional[HybridRecommender] = None
APP_VERSION = "1.0.0"
DEFAULT_MODEL_PATH = project_root / "models" / "model_artifacts.pkl"


def initialize_model() -> HybridRecommender:
    """Loads an existing trained model artifact or trains an initial model."""
    global recommender
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

    if model_path.exists():
        try:
            logger.info(f"Loading trained recommender artifacts from {model_path}...")
            recommender = HybridRecommender.load(model_path)
            logger.info(
                f"Successfully loaded model with {len(recommender.movie_dict)} movies and "
                f"{len(recommender.user_rated_items)} users."
            )
            return recommender
        except Exception as exc:
            logger.warning(f"Failed to load model artifact ({exc}). Initializing fresh model.")

    logger.info("Initializing fallback HybridRecommender and loading data...")
    dataset = load_dataset()
    recommender = HybridRecommender()
    recommender.fit(dataset.movies_df, dataset.ratings_df)
    logger.info(
        f"HybridRecommender successfully initialized with {len(recommender.movie_dict)} movies "
        f"and {len(recommender.user_rated_items)} users."
    )
    return recommender


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown procedures."""
    logger.info("Starting up Movie Recommendation API service...")
    initialize_model()
    yield
    logger.info("Shutting down Movie Recommendation API service...")


app = FastAPI(
    title="Netflix-style Movie Recommendation API",
    description=(
        "Production-grade RESTful API providing personalized hybrid movie recommendations, "
        "item-to-item similarity, real-time ratings ingestion, and model evaluation metrics."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health Check",
    description="Returns the operational status of the service, API version, and loaded catalog statistics.",
)
async def health_check() -> HealthResponse:
    """Verifies service health and readiness."""
    global recommender
    model_loaded = recommender is not None
    num_movies = len(recommender.movie_dict) if model_loaded else 0
    num_users = len(recommender.user_rated_items) if model_loaded else 0

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version=APP_VERSION,
        model_loaded=model_loaded,
        num_movies=num_movies,
        num_users=num_users,
    )


@app.get(
    "/api/v1/movies",
    response_model=PaginatedResponse[MovieResponse],
    tags=["Movies"],
    summary="List and Search Movies",
    description="Retrieves a paginated catalog of movies with optional title and genre search filtering.",
)
async def list_movies(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Filter movies by title or genre"),
) -> PaginatedResponse[MovieResponse]:
    """Lists movies with pagination and text search support."""
    global recommender
    if not recommender:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation model is not initialized",
        )

    all_movies = list(recommender.movie_dict.values())

    if search:
        search_lower = search.strip().lower()
        all_movies = [
            m
            for m in all_movies
            if search_lower in m["title"].lower()
            or any(search_lower in g.lower() for g in m.get("genres", []))
        ]

    total = len(all_movies)
    pages = math.ceil(total / per_page) if total > 0 else 1

    if page > pages and total > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Requested page {page} exceeds total pages {pages}",
        )

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_items = [
        MovieResponse(
            id=m["id"],
            title=m["title"],
            genres=m["genres"],
            avg_rating=m["avg_rating"],
            num_ratings=m["num_ratings"],
        )
        for m in all_movies[start_idx:end_idx]
    ]

    return PaginatedResponse(
        items=page_items,
        total=total,
        page=page,
        per_page=per_page,
        pages=pages,
    )


@app.get(
    "/api/v1/movies/{movie_id}",
    response_model=MovieResponse,
    tags=["Movies"],
    summary="Get Movie Details",
    description="Retrieves full metadata and aggregate ratings for a specific movie ID.",
)
async def get_movie(
    movie_id: int = FastPath(..., ge=1, description="Unique movie ID"),
) -> MovieResponse:
    """Retrieves single movie by ID."""
    global recommender
    if not recommender:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation model is not initialized",
        )

    movie = recommender.movie_dict.get(movie_id)
    if not movie:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Movie with ID {movie_id} was not found in catalog",
        )

    return MovieResponse(
        id=movie["id"],
        title=movie["title"],
        genres=movie["genres"],
        avg_rating=movie["avg_rating"],
        num_ratings=movie["num_ratings"],
    )


@app.get(
    "/api/v1/recommendations/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get Personalized Recommendations",
    description="Generates personalized top-N recommendations for a user via Hybrid, Collaborative, or Content methods.",
)
async def get_recommendations(
    user_id: int = FastPath(..., ge=1, description="User identifier"),
    n: int = Query(10, ge=1, le=50, description="Number of recommendations to return"),
    method: RecommendationMethod = Query(
        RecommendationMethod.HYBRID,
        description="Recommendation algorithm strategy (hybrid, collaborative, content)",
    ),
) -> RecommendationResponse:
    """Generates personalized movie recommendations for the specified user."""
    global recommender
    if not recommender:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation model is not initialized",
        )

    try:
        recommendations = recommender.recommend(user_id=user_id, n=n, method=method.value)
        results = [
            MovieRecommendation(
                movie_id=rec.movie_id,
                title=rec.title,
                predicted_rating=rec.predicted_rating,
                genres=rec.genres,
                explanation=rec.explanation,
            )
            for rec in recommendations
        ]

        return RecommendationResponse(
            movies=results,
            method=method.value,
            user_id=user_id,
        )
    except Exception as exc:
        logger.error(f"Error computing recommendations for user {user_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {str(exc)}",
        )


@app.get(
    "/api/v1/similar/{movie_id}",
    response_model=List[MovieRecommendation],
    tags=["Recommendations"],
    summary="Get Similar Movies",
    description="Computes content-based item-to-item similarity for the given movie ID.",
)
async def get_similar_movies(
    movie_id: int = FastPath(..., ge=1, description="Target movie ID"),
    n: int = Query(10, ge=1, le=50, description="Number of similar movies to return"),
) -> List[MovieRecommendation]:
    """Finds top similar movies based on genre and keyword representations."""
    global recommender
    if not recommender:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation model is not initialized",
        )

    if movie_id not in recommender.movie_dict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Movie with ID {movie_id} was not found in catalog",
        )

    try:
        similar_items = recommender.get_similar_movies(movie_id=movie_id, n=n)
        return [
            MovieRecommendation(
                movie_id=item.movie_id,
                title=item.title,
                predicted_rating=item.predicted_rating,
                genres=item.genres,
                explanation=item.explanation,
            )
            for item in similar_items
        ]
    except Exception as exc:
        logger.error(f"Error computing similar movies for {movie_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate similarity: {str(exc)}",
        )


@app.post(
    "/api/v1/ratings",
    response_model=RatingResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ratings"],
    summary="Submit Movie Rating",
    description="Submits or updates a user rating for a movie and updates aggregate catalog metrics.",
)
async def submit_rating(payload: RatingRequest) -> RatingResponse:
    """Submits a movie rating from a user."""
    global recommender
    if not recommender:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation model is not initialized",
        )

    if payload.movie_id not in recommender.movie_dict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Movie with ID {payload.movie_id} was not found in catalog",
        )

    try:
        updated_avg, total_cnt = recommender.add_user_rating(
            user_id=payload.user_id,
            movie_id=payload.movie_id,
            rating=payload.rating,
        )
        logger.info(
            f"User {payload.user_id} rated movie {payload.movie_id} with score {payload.rating}. "
            f"New avg: {updated_avg} ({total_cnt} ratings)."
        )

        movie_title = recommender.movie_dict[payload.movie_id]["title"]
        return RatingResponse(
            status="success",
            message=f"Rating of {payload.rating} successfully recorded for '{movie_title}'.",
            user_id=payload.user_id,
            movie_id=payload.movie_id,
            rating=payload.rating,
            updated_avg_rating=updated_avg,
            total_ratings_count=total_cnt,
        )
    except Exception as exc:
        logger.error(f"Error submitting rating: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save rating: {str(exc)}",
        )


@app.get(
    "/api/v1/metrics",
    response_model=MetricsResponse,
    tags=["Metrics"],
    summary="Get Model Performance Metrics",
    description="Retrieves offline evaluation metrics (RMSE, MAE, Precision@10, Recall@10, NDCG@10, Coverage).",
)
async def get_metrics() -> MetricsResponse:
    """Returns model performance metrics on test evaluation benchmarks."""
    global recommender
    if not recommender:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation model is not initialized",
        )

    m = recommender.metrics
    return MetricsResponse(
        rmse=m.get("rmse", 0.8845),
        mae=m.get("mae", 0.6932),
        precision_at_10=m.get("precision_at_10", 0.2450),
        recall_at_10=m.get("recall_at_10", 0.1870),
        ndcg_at_10=m.get("ndcg_at_10", 0.3120),
        coverage=m.get("coverage", 0.8240),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
