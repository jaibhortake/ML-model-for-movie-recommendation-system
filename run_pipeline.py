"""End-to-end execution script for the Netflix-style hybrid movie recommendation pipeline."""

import logging
import sys
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.data.loader import load_movies, load_ratings
from src.data.preprocessor import train_test_split
from src.models.collaborative import MatrixFactorizationALS
from src.models.content_based import ContentBasedReconmender if False else None  # type check stub
from src.models.content_based import ContentBasedRecommender
from src.models.evaluation import evaluate_model
from src.models.hybrid import HybridRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("PipelineRunner")


def run_pipeline():
    logger.info("=== STEP 1: LOADING MOVIELENS 100K DATASET ===")
    ratings_df = load_ratings(data_dir="data/raw")
    movies_df = load_movies(data_dir="data/raw")

    logger.info("=== STEP 2: TEMPORAL TRAIN/TEST SPLIT (80/20) ===")
    train_df, test_df = train_test_split(ratings_df, test_ratio=0.2, temporal=True)

    logger.info("=== STEP 3: HYBRID RECOMMENDER TRAINING ===")
    hybrid = HybridRecommender(
        cf_weight=0.7,
        cb_weight=0.3,
        cold_start_threshold=5,
        n_factors=50,
        n_iterations=15,
        reg=0.1,
    )
    hybrid.fit(ratings_df=train_df, movies_df=movies_df)

    logger.info("=== STEP 4: RECOMMENDATION & EXPLANATION FOR SAMPLE USER ===")
    sample_user_id = int(train_df["user_id"].iloc[0])
    logger.info("Generating Top-5 recommendations for user_id=%d", sample_user_id)
    recs = hybrid.recommend(user_id=sample_user_id, n=5)
    for rank, r in enumerate(recs, start=1):
        logger.info(
            "Rank %d: '%s' (Score: %.3f | CF: %.3f | CB: %.3f | Strategy: %s)",
            rank,
            r["title"],
            r["score"],
            r["cf_score"],
            r["cb_score"],
            r["strategy"],
        )

    if recs:
        top_movie_id = recs[0]["movie_id"]
        explanation = hybrid.get_explanation(user_id=sample_user_id, movie_id=top_movie_id)
        logger.info("Explanation for top recommendation:")
        logger.info("  Movie: %s", explanation["title"])
        logger.info("  Reason: %s", explanation["reason"])
        logger.info("  Similar Liked Films: %s", explanation["similar_liked_movies"])

    logger.info("=== STEP 5: MODEL EVALUATION ON HELD-OUT TEST SET ===")
    eval_metrics = evaluate_model(
        model=hybrid.cf_model,
        test_data=test_df,
        train_data=train_df,
        k=10,
        max_eval_users=50,
    )
    logger.info("Collaborative Filtering Benchmark Metrics:")
    for metric, value in eval_metrics.items():
        logger.info("  %s: %s", metric, value)

    # Save fitted model
    artifacts_dir = Path("artifacts/models")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    hybrid.save(artifacts_dir / "hybrid_model.joblib")
    logger.info("Successfully exported hybrid model to %s", artifacts_dir / "hybrid_model.joblib")


if __name__ == "__main__":
    run_pipeline()
