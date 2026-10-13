"""Dataset management, downloading, and preprocessing for Movie Recommendation System."""

from __future__ import annotations

import io
import logging
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("movie_rec_data")

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

DEFAULT_GENRES = [
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

SEED_MOVIES = [
    (1, "Toy Story (1995)", ["Animation", "Children's", "Comedy"]),
    (2, "GoldenEye (1995)", ["Action", "Adventure", "Thriller"]),
    (3, "Four Rooms (1995)", ["Comedy", "Thriller"]),
    (4, "Get Shorty (1995)", ["Action", "Comedy", "Drama"]),
    (5, "Copycat (1995)", ["Crime", "Drama", "Thriller"]),
    (6, "Shanghai Triad (1995)", ["Drama"]),
    (7, "Twelve Monkeys (1995)", ["Drama", "Sci-Fi"]),
    (8, "Babe (1995)", ["Children's", "Comedy", "Drama"]),
    (9, "Dead Man Walking (1995)", ["Drama"]),
    (10, "Richard III (1995)", ["Drama", "War"]),
    (11, "Seven (1995)", ["Crime", "Thriller"]),
    (12, "Usual Suspects, The (1995)", ["Crime", "Thriller"]),
    (22, "Braveheart (1995)", ["Action", "Drama", "War"]),
    (28, "Apollo 13 (1995)", ["Action", "Drama", "Thriller"]),
    (50, "Star Wars: Episode IV - A New Hope (1977)", ["Action", "Adventure", "Romance", "Sci-Fi", "War"]),
    (56, "Pulp Fiction (1994)", ["Crime", "Drama"]),
    (64, "Shawshank Redemption, The (1994)", ["Drama"]),
    (69, "Forrest Gump (1994)", ["Comedy", "Romance", "War"]),
    (79, "Fugitive, The (1993)", ["Action", "Thriller"]),
    (82, "Jurassic Park (1993)", ["Action", "Adventure", "Sci-Fi"]),
    (89, "Blade Runner (1982)", ["Film-Noir", "Sci-Fi"]),
    (96, "Terminator 2: Judgment Day (1991)", ["Action", "Sci-Fi", "Thriller"]),
    (98, "Silence of the Lambs, The (1991)", ["Drama", "Thriller"]),
    (100, "Fargo (1996)", ["Crime", "Drama", "Thriller"]),
    (121, "Independence Day (ID4) (1996)", ["Action", "Sci-Fi", "War"]),
    (127, "Godfather, The (1972)", ["Action", "Crime", "Drama"]),
    (133, "Gone with the Wind (1939)", ["Drama", "Romance", "War"]),
    (144, "Die Hard (1988)", ["Action", "Thriller"]),
    (168, "Monty Python and the Holy Grail (1975)", ["Comedy"]),
    (172, "Empire Strikes Back, The (1980)", ["Action", "Adventure", "Drama", "Romance", "Sci-Fi", "War"]),
    (173, "Princess Bride, The (1987)", ["Action", "Adventure", "Comedy", "Romance"]),
    (174, "Raiders of the Lost Ark (1981)", ["Action", "Adventure"]),
    (181, "Return of the Jedi (1983)", ["Action", "Adventure", "Romance", "Sci-Fi", "War"]),
    (187, "Godfather: Part II, The (1974)", ["Action", "Crime", "Drama"]),
    (195, "Terminator, The (1984)", ["Action", "Sci-Fi", "Thriller"]),
    (196, "Dead Poets Society (1989)", ["Drama"]),
    (204, "Back to the Future (1985)", ["Comedy", "Sci-Fi"]),
    (210, "Indiana Jones and the Last Crusade (1989)", ["Action", "Adventure"]),
    (222, "Star Trek: First Contact (1996)", ["Action", "Adventure", "Sci-Fi"]),
    (228, "Finding Nemo (2003)", ["Animation", "Children's", "Comedy"]),
    (238, "Raising Arizona (1987)", ["Comedy"]),
    (257, "Men in Black (1997)", ["Action", "Adventure", "Comedy", "Sci-Fi"]),
    (258, "Contact (1997)", ["Drama", "Sci-Fi"]),
    (260, "Star Wars: Episode IV - A New Hope (Special Edition) (1997)", ["Action", "Adventure", "Sci-Fi"]),
    (275, "Sense and Sensibility (1995)", ["Drama", "Romance"]),
    (288, "Scream (1996)", ["Horror", "Thriller"]),
    (294, "Liar Liar (1997)", ["Comedy"]),
    (300, "Air Force One (1997)", ["Action", "Thriller"]),
    (302, "L.A. Confidential (1997)", ["Crime", "Film-Noir", "Mystery", "Thriller"]),
    (313, "Titanic (1997)", ["Action", "Drama", "Romance"]),
    (318, "Schindler's List (1993)", ["Drama", "War"]),
    (357, "Matrix, The (1999)", ["Action", "Sci-Fi", "Thriller"]),
    (403, "Batman (1989)", ["Action", "Adventure", "Crime", "Drama"]),
    (405, "Mission: Impossible (1996)", ["Action", "Adventure", "Mystery"]),
    (423, "E.T. the Extra-Terrestrial (1982)", ["Children's", "Drama", "Fantasy", "Sci-Fi"]),
    (508, "Inception (2010)", ["Action", "Crime", "Sci-Fi", "Thriller"]),
    (510, "Dark Knight, The (2008)", ["Action", "Crime", "Drama"]),
    (515, "Fight Club (1999)", ["Drama", "Thriller"]),
    (527, "Interstellar (2014)", ["Adventure", "Drama", "Sci-Fi"]),
    (568, "Spirited Away (2001)", ["Animation", "Adventure", "Fantasy"]),
]


@dataclass
class DatasetBundle:
    """Container holding loaded movie catalog and ratings datasets."""

    movies_df: pd.DataFrame
    ratings_df: pd.DataFrame
    movie_dict: Dict[int, dict]


def generate_synthetic_dataset(
    num_users: int = 300,
    min_ratings_per_user: int = 15,
    max_ratings_per_user: int = 40,
    seed: int = 42,
) -> DatasetBundle:
    """Generates a realistic synthetic MovieLens-compatible dataset.

    Args:
        num_users: Number of simulated users.
        min_ratings_per_user: Minimum ratings per user.
        max_ratings_per_user: Maximum ratings per user.
        seed: Random seed for reproducibility.

    Returns:
        DatasetBundle containing movies_df, ratings_df, and movie_dict.
    """
    rng = np.random.RandomState(seed)

    movies_records = []
    movie_dict = {}
    for movie_id, title, genres in SEED_MOVIES:
        record = {
            "movie_id": movie_id,
            "title": title,
            "genres": genres,
            "genres_str": "|".join(genres),
        }
        movies_records.append(record)
        movie_dict[movie_id] = {
            "id": movie_id,
            "title": title,
            "genres": genres,
            "avg_rating": 3.5,
            "num_ratings": 0,
        }

    movies_df = pd.DataFrame(movies_records)
    all_movie_ids = movies_df["movie_id"].values

    ratings_rows = []
    # User latent genre preferences
    genre_preferences = {
        u: rng.dirichlet(np.ones(len(DEFAULT_GENRES)) * 1.5) for u in range(1, num_users + 1)
    }

    for user_id in range(1, num_users + 1):
        num_ratings = rng.randint(min_ratings_per_user, max_ratings_per_user + 1)
        selected_movies = rng.choice(all_movie_ids, size=min(num_ratings, len(all_movie_ids)), replace=False)

        for m_id in selected_movies:
            # Base quality of movie (some movies are higher quality)
            movie_genres = movie_dict[m_id]["genres"]
            user_pref = sum(
                genre_preferences[user_id][DEFAULT_GENRES.index(g)]
                for g in movie_genres
                if g in DEFAULT_GENRES
            ) / max(1, len(movie_genres))

            base_score = 3.2 + (user_pref * 2.5) + rng.normal(0, 0.4)
            # Clip between 1.0 and 5.0 in 0.5 increments
            rating = float(np.clip(np.round(base_score * 2) / 2, 1.0, 5.0))
            ratings_rows.append({"user_id": user_id, "movie_id": int(m_id), "rating": rating})

    ratings_df = pd.DataFrame(ratings_rows)

    # Compute aggregate stats
    stats = ratings_df.groupby("movie_id")["rating"].agg(["mean", "count"]).reset_index()
    stats_dict = {
        int(row["movie_id"]): (round(float(row["mean"]), 2), int(row["count"]))
        for _, row in stats.iterrows()
    }

    for m_id, item in movie_dict.items():
        if m_id in stats_dict:
            item["avg_rating"] = stats_dict[m_id][0]
            item["num_ratings"] = stats_dict[m_id][1]

    movies_df["avg_rating"] = movies_df["movie_id"].map(
        lambda mid: movie_dict.get(mid, {}).get("avg_rating", 3.5)
    )
    movies_df["num_ratings"] = movies_df["movie_id"].map(
        lambda mid: movie_dict.get(mid, {}).get("num_ratings", 0)
    )

    return DatasetBundle(movies_df=movies_df, ratings_df=ratings_df, movie_dict=movie_dict)


def load_dataset(data_dir: Optional[str] = None, auto_download: bool = False) -> DatasetBundle:
    """Loads MovieLens data or returns synthetic dataset.

    Args:
        data_dir: Optional local directory where MovieLens files exist.
        auto_download: Whether to attempt downloading MovieLens 100k if not found.

    Returns:
        DatasetBundle containing movies and ratings.
    """
    if data_dir and Path(data_dir).exists():
        path = Path(data_dir)
        u_data = path / "u.data"
        u_item = path / "u.item"
        if u_data.exists() and u_item.exists():
            try:
                logger.info(f"Loading MovieLens 100k dataset from {data_dir}")
                ratings_df = pd.read_csv(
                    u_data,
                    sep="\t",
                    names=["user_id", "movie_id", "rating", "timestamp"],
                    usecols=["user_id", "movie_id", "rating"],
                )
                genre_cols = [
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
                item_cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + genre_cols
                movies_raw = pd.read_csv(
                    u_item,
                    sep="|",
                    names=item_cols,
                    encoding="latin-1",
                    usecols=["movie_id", "title"] + genre_cols,
                )

                def extract_genres(row):
                    active = [g for g in genre_cols if g != "unknown" and row[g] == 1]
                    return active if active else ["Drama"]

                movies_raw["genres"] = movies_raw.apply(extract_genres, axis=1)
                movies_raw["genres_str"] = movies_raw["genres"].apply(lambda g: "|".join(g))

                movie_dict = {}
                stats = ratings_df.groupby("movie_id")["rating"].agg(["mean", "count"]).reset_index()
                stats_dict = {
                    int(row["movie_id"]): (round(float(row["mean"]), 2), int(row["count"]))
                    for _, row in stats.iterrows()
                }

                for _, row in movies_raw.iterrows():
                    m_id = int(row["movie_id"])
                    avg_r, cnt = stats_dict.get(m_id, (3.5, 0))
                    movie_dict[m_id] = {
                        "id": m_id,
                        "title": str(row["title"]),
                        "genres": row["genres"],
                        "avg_rating": avg_r,
                        "num_ratings": cnt,
                    }

                movies_df = movies_raw[["movie_id", "title", "genres", "genres_str"]].copy()
                movies_df["avg_rating"] = movies_df["movie_id"].map(lambda mid: movie_dict[mid]["avg_rating"])
                movies_df["num_ratings"] = movies_df["movie_id"].map(lambda mid: movie_dict[mid]["num_ratings"])

                return DatasetBundle(movies_df=movies_df, ratings_df=ratings_df, movie_dict=movie_dict)
            except Exception as e:
                logger.warning(f"Failed to load MovieLens from {data_dir}: {e}. Falling back to synthetic.")

    logger.info("Using embedded rich dataset for movies and ratings.")
    return generate_synthetic_dataset()
