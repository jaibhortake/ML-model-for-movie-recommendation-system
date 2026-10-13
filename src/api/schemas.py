"""Pydantic schemas and data validation models for the Movie Recommendation System API."""

from __future__ import annotations

from enum import Enum
from typing import Generic, List, Optional, TypeVar
from pydantic import BaseModel, ConfigDict, Field


class RecommendationMethod(str, Enum):
    """Supported recommendation methods."""

    HYBRID = "hybrid"
    COLLABORATIVE = "collaborative"
    CONTENT = "content"


class MovieResponse(BaseModel):
    """Schema representing movie details and aggregate statistics."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Unique movie identifier", examples=[1])
    title: str = Field(..., description="Full movie title with release year", examples=["Toy Story (1995)"])
    genres: List[str] = Field(..., description="List of movie genres", examples=[["Animation", "Children's", "Comedy"]])
    avg_rating: float = Field(..., description="Average rating score across all users (1.0 to 5.0)", examples=[4.15])
    num_ratings: int = Field(..., description="Total count of user ratings received", examples=[2077])


class MovieRecommendation(BaseModel):
    """Schema for an individual movie recommendation."""

    model_config = ConfigDict(from_attributes=True)

    movie_id: int = Field(..., description="Unique movie identifier", examples=[260])
    title: str = Field(..., description="Full movie title", examples=["Star Wars: Episode IV - A New Hope (1977)"])
    predicted_rating: float = Field(..., description="Predicted rating score (1.0 to 5.0)", examples=[4.68])
    genres: List[str] = Field(..., description="List of genres associated with the movie", examples=[["Action", "Adventure", "Sci-Fi"]])
    explanation: str = Field(
        ...,
        description="Human-readable justification for the recommendation",
        examples=["Recommended because you rated Sci-Fi and Adventure classics highly."],
    )


class RecommendationResponse(BaseModel):
    """Schema for personalized recommendations response."""

    model_config = ConfigDict(from_attributes=True)

    movies: List[MovieRecommendation] = Field(..., description="Ranked list of recommended movies")
    method: str = Field(..., description="Recommendation algorithm utilized", examples=["hybrid"])
    user_id: int = Field(..., description="Target user identifier", examples=[42])


class RatingRequest(BaseModel):
    """Schema for submitting a user rating on a movie."""

    model_config = ConfigDict(from_attributes=True)

    user_id: int = Field(..., ge=1, description="ID of the user submitting the rating", examples=[42])
    movie_id: int = Field(..., ge=1, description="ID of the movie being rated", examples=[1])
    rating: float = Field(
        ...,
        ge=1.0,
        le=5.0,
        description="Rating value on a 1.0 to 5.0 scale (increments of 0.5 or integers)",
        examples=[4.5],
    )


class RatingResponse(BaseModel):
    """Schema for rating submission confirmation."""

    model_config = ConfigDict(from_attributes=True)

    status: str = Field(default="success", description="Status of the operation")
    message: str = Field(..., description="Detailed result message")
    user_id: int = Field(..., description="User ID")
    movie_id: int = Field(..., description="Movie ID")
    rating: float = Field(..., description="Recorded rating")
    updated_avg_rating: Optional[float] = Field(None, description="New average rating for the movie")
    total_ratings_count: Optional[int] = Field(None, description="Updated total count of ratings for the movie")


T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic pagination wrapper for list endpoints."""

    model_config = ConfigDict(from_attributes=True)

    items: List[T] = Field(..., description="Page items")
    total: int = Field(..., description="Total items available", examples=[1682])
    page: int = Field(..., description="Current page number (1-indexed)", examples=[1])
    per_page: int = Field(..., description="Number of items per page", examples=[20])
    pages: int = Field(..., description="Total number of pages", examples=[85])


class HealthResponse(BaseModel):
    """Schema for API health status check."""

    model_config = ConfigDict(from_attributes=True)

    status: str = Field(..., description="System status indicator ('healthy' or 'degraded')", examples=["healthy"])
    version: str = Field(..., description="API semantic version", examples=["1.0.0"])
    model_loaded: bool = Field(..., description="Whether the recommender model is loaded in memory", examples=[True])
    num_movies: int = Field(..., description="Count of indexed movies in the catalog", examples=[1682])
    num_users: int = Field(..., description="Count of indexed users in the system", examples=[943])


class MetricsResponse(BaseModel):
    """Schema for model evaluation and offline performance metrics."""

    model_config = ConfigDict(from_attributes=True)

    rmse: float = Field(..., description="Root Mean Squared Error on holdout test set", examples=[0.8845])
    mae: float = Field(..., description="Mean Absolute Error on holdout test set", examples=[0.6932])
    precision_at_10: float = Field(..., description="Top-10 Precision recommendation quality metric", examples=[0.2450])
    recall_at_10: float = Field(..., description="Top-10 Recall recommendation coverage metric", examples=[0.1870])
    ndcg_at_10: float = Field(..., description="Normalized Discounted Cumulative Gain at rank 10", examples=[0.3120])
    coverage: float = Field(..., description="Catalog coverage percentage (0.0 to 1.0)", examples=[0.8240])


class SimilarMoviesResponse(BaseModel):
    """Schema for item-to-item similarity recommendations."""

    model_config = ConfigDict(from_attributes=True)

    movie_id: int = Field(..., description="Source movie identifier", examples=[1])
    title: str = Field(..., description="Source movie title", examples=["Toy Story (1995)"])
    similar_movies: List[MovieRecommendation] = Field(..., description="List of most similar movies")
