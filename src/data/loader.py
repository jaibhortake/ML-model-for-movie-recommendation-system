"""Data loading module for MovieLens 100K dataset.

Provides automated download, caching, extraction, and parsing of ratings,
movies, and genre metadata from GroupLens MovieLens 100K dataset.
"""

from __future__ import annotations

import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MOVIELENS_URL: Final[str] = (
    "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
)

GENRE_COLUMNS: Final[list[str]] = [
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

ITEM_COLUMNS: Final[list[str]] = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
] + GENRE_COLUMNS


def download_movielens(
    data_dir: Path | str = "data/raw",
    url: str = DEFAULT_MOVIELENS_URL,
    force_download: bool = False,
) -> Path:
    """Download and extract MovieLens 100K dataset if not already cached.

    Args:
        data_dir: Target directory where data should be extracted and cached.
        url: Remote URL pointing to the ml-100k.zip dataset archive.
        force_download: If True, re-downloads even if cached files exist.

    Returns:
        Path to the extracted `ml-100k` directory containing dataset files.

    Raises:
        RuntimeError: If download or extraction fails.
    """
    base_path = Path(data_dir).resolve()
    target_dir = base_path / "ml-100k"
    u_data_path = target_dir / "u.data"
    u_item_path = target_dir / "u.item"

    if not force_download and u_data_path.is_file() and u_item_path.is_file():
        logger.info("Found cached MovieLens 100K dataset at %s", target_dir)
        return target_dir

    base_path.mkdir(parents=True, exist_ok=True)
    zip_path = base_path / "ml-100k.zip"

    try:
        logger.info("Downloading MovieLens 100K from %s to %s...", url, zip_path)
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0 (MovieRecML/1.0)")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, zip_path)

        logger.info("Extracting %s to %s...", zip_path, base_path)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(base_path)

        if not target_dir.exists():
            # Check if nested or named differently
            extracted_dirs = [p for p in base_path.iterdir() if p.is_dir() and "100k" in p.name.lower()]
            if extracted_dirs:
                if extracted_dirs[0] != target_dir:
                    shutil.move(str(extracted_dirs[0]), str(target_dir))

        logger.info("Dataset successfully extracted to %s", target_dir)
    except Exception as exc:
        logger.error("Failed to download or extract MovieLens 100K: %s", exc)
        raise RuntimeError(f"MovieLens 100K download/extract error: {exc}") from exc
    finally:
        if zip_path.exists():
            try:
                zip_path.unlink()
            except OSError:
                pass

    return target_dir


def load_ratings(
    data_dir: Path | str = "data/raw",
    auto_download: bool = True,
) -> pd.DataFrame:
    """Load ratings data from u.data file.

    Args:
        data_dir: Base directory for the raw dataset.
        auto_download: Automatically download the dataset if missing.

    Returns:
        DataFrame containing columns: `user_id`, `item_id`, `rating`, `timestamp`, `datetime`.

    Raises:
        FileNotFoundError: If `u.data` cannot be found and `auto_download` is False.
    """
    raw_dir = Path(data_dir).resolve()
    target_dir = raw_dir / "ml-100k" if raw_dir.name != "ml-100k" else raw_dir
    u_data_file = target_dir / "u.data"

    if not u_data_file.exists():
        if auto_download:
            target_dir = download_movielens(data_dir=raw_dir)
            u_data_file = target_dir / "u.data"
        else:
            raise FileNotFoundError(f"Ratings file not found at {u_data_file}")

    logger.info("Loading ratings from %s", u_data_file)
    ratings_df = pd.read_csv(
        u_data_file,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="c",
        dtype={
            "user_id": np.int32,
            "item_id": np.int32,
            "rating": np.float32,
            "timestamp": np.int64,
        },
    )

    ratings_df["datetime"] = pd.to_datetime(ratings_df["timestamp"], unit="s")
    logger.info(
        "Loaded %d ratings across %d users and %d items. Sparsity: %.2f%%",
        len(ratings_df),
        ratings_df["user_id"].nunique(),
        ratings_df["item_id"].nunique(),
        100.0 * (1.0 - len(ratings_df) / (ratings_df["user_id"].nunique() * ratings_df["item_id"].nunique())),
    )
    return ratings_df


def load_genres(
    data_dir: Path | str = "data/raw",
    auto_download: bool = True,
) -> pd.DataFrame:
    """Load genre names and IDs from u.genre file.

    Args:
        data_dir: Base directory for the raw dataset.
        auto_download: Automatically download the dataset if missing.

    Returns:
        DataFrame containing columns: `genre`, `genre_id`.
    """
    raw_dir = Path(data_dir).resolve()
    target_dir = raw_dir / "ml-100k" if raw_dir.name != "ml-100k" else raw_dir
    u_genre_file = target_dir / "u.genre"

    if not u_genre_file.exists():
        if auto_download:
            target_dir = download_movielens(data_dir=raw_dir)
            u_genre_file = target_dir / "u.genre"
        else:
            raise FileNotFoundError(f"Genre file not found at {u_genre_file}")

    genres_df = pd.read_csv(
        u_genre_file,
        sep="|",
        names=["genre", "genre_id"],
        encoding="latin-1",
        engine="python",
    ).dropna()

    genres_df["genre_id"] = genres_df["genre_id"].astype(np.int32)
    return genres_df


def load_movies(
    data_dir: Path | str = "data/raw",
    auto_download: bool = True,
) -> pd.DataFrame:
    """Load movie metadata from u.item file.

    Args:
        data_dir: Base directory for the raw dataset.
        auto_download: Automatically download the dataset if missing.

    Returns:
        DataFrame containing movie id, title, release date, URLs, and genre binary flags,
        along with an aggregated `genres` list column.
    """
    raw_dir = Path(data_dir).resolve()
    target_dir = raw_dir / "ml-100k" if raw_dir.name != "ml-100k" else raw_dir
    u_item_file = target_dir / "u.item"

    if not u_item_file.exists():
        if auto_download:
            target_dir = download_movielens(data_dir=raw_dir)
            u_item_file = target_dir / "u.item"
        else:
            raise FileNotFoundError(f"Movie metadata file not found at {u_item_file}")

    logger.info("Loading movies from %s", u_item_file)
    movies_df = pd.read_csv(
        u_item_file,
        sep="|",
        names=ITEM_COLUMNS,
        encoding="latin-1",
        engine="python",
    )

    movies_df["movie_id"] = movies_df["movie_id"].astype(np.int32)

    # Derive genres list column from binary flags
    def _extract_genre_list(row: pd.Series) -> list[str]:
        return [g for g in GENRE_COLUMNS if row.get(g, 0) == 1]

    movies_df["genre_list"] = movies_df.apply(_extract_genre_list, axis=1)
    movies_df["genres_str"] = movies_df["genre_list"].apply(lambda lst: "|".join(lst) if lst else "unknown")

    logger.info("Loaded %d movies with genre annotations.", len(movies_df))
    return movies_df


def build_user_item_matrix(
    ratings_df: pd.DataFrame,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Build a dense 2D user-item matrix from ratings DataFrame.

    Args:
        ratings_df: DataFrame containing at least `user_id`, `item_id`, and `rating`.
        fill_value: Value used to represent missing/unrated user-item pairs (defaults to 0.0).

    Returns:
        Pivot DataFrame with users as rows (index) and item IDs as columns.
    """
    required_cols = {"user_id", "item_id", "rating"}
    if not required_cols.issubset(ratings_df.columns):
        raise ValueError(f"ratings_df must contain {required_cols}, found {ratings_df.columns.tolist()}")

    logger.info("Building user-item matrix from %d ratings...", len(ratings_df))
    matrix = ratings_df.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating",
        fill_value=fill_value,
    )
    logger.info(
        "User-item matrix constructed with shape (users=%d, items=%d)",
        matrix.shape[0],
        matrix.shape[1],
    )
    return matrix
