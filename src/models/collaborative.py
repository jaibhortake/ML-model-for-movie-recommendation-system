"""Collaborative filtering models for movie recommendation.

Implements:
1. MatrixFactorizationALS: Alternating Least Squares with numpy and scipy.
2. NeuralCollaborativeFilter: NeuMF (Generalized Matrix Factorization + Deep MLP)
   using TensorFlow / Keras with embedding representations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

logger = logging.getLogger(__name__)


class MatrixFactorizationALS:
    """Alternating Least Squares (ALS) Matrix Factorization recommender.

    Decomposes the user-item rating matrix R into the product of low-rank
    user latent factors U and item latent factors V:
        R ≈ U @ V.T

    Optimizes using exact ridge least squares alternating updates:
        u_u = (V_I.T @ V_I + reg * I)^(-1) @ V_I.T @ r_u
        v_i = (U_I.T @ U_I + reg * I)^(-1) @ U_I.T @ r_i
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_iterations: int = 20,
        reg: float = 0.1,
        random_state: int = 42,
    ) -> None:
        """Initialize the ALS model hyper-parameters.

        Args:
            n_factors: Dimension of latent factor vectors.
            n_iterations: Number of alternating user/item coordinate iterations.
            reg: L2 regularization hyperparameter lambda.
            random_state: Seed for random latent initialization.
        """
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg = reg
        self.random_state = random_state

        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.global_mean: float = 0.0
        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}
        self.ratings_sparse: csr_matrix | None = None

    def fit(
        self,
        ratings_matrix: np.ndarray | csr_matrix | pd.DataFrame,
        n_factors: int | None = None,
        n_iterations: int | None = None,
        reg: float | None = None,
        user_to_idx: dict[int, int] | None = None,
        item_to_idx: dict[int, int] | None = None,
    ) -> MatrixFactorizationALS:
        """Fit ALS latent factor matrices on the user-item interaction matrix.

        Args:
            ratings_matrix: User-item matrix. Can be a dense 2D ndarray, SciPy
                csr_matrix, or Pandas DataFrame (with user index and item columns).
            n_factors: Override latent dimensions if specified.
            n_iterations: Override iteration count if specified.
            reg: Override L2 regularization if specified.
            user_to_idx: Optional dictionary mapping user_id to row index.
            item_to_idx: Optional dictionary mapping movie_id to column index.

        Returns:
            Fitted instance of MatrixFactorizationALS.
        """
        if n_factors is not None:
            self.n_factors = n_factors
        if n_iterations is not None:
            self.n_iterations = n_iterations
        if reg is not None:
            self.reg = reg

        # Convert input matrix into CSR sparse representation
        if isinstance(ratings_matrix, pd.DataFrame):
            if user_to_idx is None:
                self.user_to_idx = {uid: i for i, uid in enumerate(ratings_matrix.index)}
            else:
                self.user_to_idx = user_to_idx
            if item_to_idx is None:
                self.item_to_idx = {mid: j for j, mid in enumerate(ratings_matrix.columns)}
            else:
                self.item_to_idx = item_to_idx
            sparse_mat = csr_matrix(ratings_matrix.to_numpy(dtype=np.float32))
        elif issparse(ratings_matrix):
            sparse_mat = ratings_matrix.tocsr().astype(np.float32)
            if user_to_idx is not None:
                self.user_to_idx = user_to_idx
            else:
                self.user_to_idx = {i: i for i in range(sparse_mat.shape[0])}
            if item_to_idx is not None:
                self.item_to_idx = item_to_idx
            else:
                self.item_to_idx = {j: j for j in range(sparse_mat.shape[1])}
        else:
            sparse_mat = csr_matrix(ratings_matrix, dtype=np.float32)
            self.user_to_idx = user_to_idx or {i: i for i in range(sparse_mat.shape[0])}
            self.item_to_idx = item_to_idx or {j: j for j in range(sparse_mat.shape[1])}

        self.idx_to_user = {i: uid for uid, i in self.user_to_idx.items()}
        self.idx_to_item = {j: mid for mid, j in self.item_to_idx.items()}
        self.ratings_sparse = sparse_mat

        n_users, n_items = sparse_mat.shape
        non_zeros = sparse_mat.data
        self.global_mean = float(np.mean(non_zeros)) if len(non_zeros) > 0 else 0.0

        rng = np.random.default_rng(self.random_state)
        # Latent factor initialization scaled by variance
        scale = 1.0 / np.sqrt(self.n_factors)
        self.user_factors = rng.normal(0, scale, size=(n_users, self.n_factors)).astype(np.float32)
        self.item_factors = rng.normal(0, scale, size=(n_items, self.n_factors)).astype(np.float32)

        # Precompute transposed CSR for column-slicing efficiency
        ratings_csc = sparse_mat.tocsc()
        lambda_i = self.reg * np.eye(self.n_factors, dtype=np.float32)

        logger.info(
            "Starting ALS fit: %d users, %d items, %d factors, %d iterations, reg=%.3f",
            n_users,
            n_items,
            self.n_factors,
            self.n_iterations,
            self.reg,
        )

        for it in range(1, self.n_iterations + 1):
            # 1. Update User Factors
            for u in range(n_users):
                item_indices = sparse_mat.indices[sparse_mat.indptr[u] : sparse_mat.indptr[u + 1]]
                if len(item_indices) == 0:
                    continue
                v_subset = self.item_factors[item_indices]
                r_subset = sparse_mat.data[sparse_mat.indptr[u] : sparse_mat.indptr[u + 1]]

                a_matrix = v_subset.T @ v_subset + lambda_i
                b_vector = v_subset.T @ r_subset
                try:
                    self.user_factors[u] = np.linalg.solve(a_matrix, b_vector)
                except np.linalg.LinAlgError:
                    self.user_factors[u] = np.linalg.lstsq(a_matrix, b_vector, rcond=None)[0]

            # 2. Update Item Factors
            for i in range(n_items):
                user_indices = ratings_csc.indices[ratings_csc.indptr[i] : ratings_csc.indptr[i + 1]]
                if len(user_indices) == 0:
                    continue
                u_subset = self.user_factors[user_indices]
                r_subset = ratings_csc.data[ratings_csc.indptr[i] : ratings_csc.indptr[i + 1]]

                a_matrix = u_subset.T @ u_subset + lambda_i
                b_vector = u_subset.T @ r_subset
                try:
                    self.item_factors[i] = np.linalg.solve(a_matrix, b_vector)
                except np.linalg.LinAlgError:
                    self.item_factors[i] = np.linalg.lstsq(a_matrix, b_vector, rcond=None)[0]

            # Log loss monitoring every 5 iterations or last
            if it % 5 == 0 or it == self.n_iterations:
                loss = self._compute_training_loss(sparse_mat)
                logger.info("ALS Iteration %d/%d completed - Training RMSE: %.4f", it, self.n_iterations, loss)

        logger.info("ALS model training successfully finished.")
        return self

    def _compute_training_loss(self, sparse_mat: csr_matrix) -> float:
        """Compute RMSE on observed ratings."""
        if self.user_factors is None or self.item_factors is None:
            return float("nan")
        users, items = sparse_mat.nonzero()
        preds = np.sum(self.user_factors[users] * self.item_factors[items], axis=1)
        actuals = sparse_mat.data
        mse = np.mean((actuals - preds) ** 2)
        return float(np.sqrt(mse))

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the rating score for a single user and movie pair.

        Args:
            user_id: Target user ID.
            movie_id: Target movie ID.

        Returns:
            Predicted rating bounded between 1.0 and 5.0.
        """
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        u_idx = self.user_to_idx.get(user_id)
        i_idx = self.item_to_idx.get(movie_id)

        if u_idx is None and i_idx is None:
            return self.global_mean
        if u_idx is None:
            # Fallback to item average if known
            if self.ratings_sparse is not None and i_idx is not None:
                item_ratings = self.ratings_sparse.getcol(i_idx).data
                return float(np.mean(item_ratings)) if len(item_ratings) > 0 else self.global_mean
            return self.global_mean
        if i_idx is None:
            # Fallback to user average
            if self.ratings_sparse is not None and u_idx is not None:
                user_ratings = self.ratings_sparse.getrow(u_idx).data
                return float(np.mean(user_ratings)) if len(user_ratings) > 0 else self.global_mean
            return self.global_mean

        pred = float(np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True,
    ) -> list[tuple[int, float]]:
        """Generate top-N movie recommendations for a specific user.

        Args:
            user_id: Target user ID.
            n: Number of recommendations to produce.
            exclude_rated: If True, filters out items already rated by user.

        Returns:
            List of (movie_id, predicted_score) tuples sorted in descending order.
        """
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        u_idx = self.user_to_idx.get(user_id)
        if u_idx is None:
            logger.warning("Cold-start user %d: falling back to highest average rated movies", user_id)
            return self._cold_start_recommendations(n)

        # Compute inner product against all item latent representations
        scores = self.user_factors[u_idx] @ self.item_factors.T

        if exclude_rated and self.ratings_sparse is not None:
            rated_indices = self.ratings_sparse.indices[
                self.ratings_sparse.indptr[u_idx] : self.ratings_sparse.indptr[u_idx + 1]
            ]
            scores[rated_indices] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n]
        recommendations: list[tuple[int, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score == -np.inf:
                break
            movie_id = self.idx_to_item.get(idx, int(idx))
            clipped_score = float(np.clip(score, 1.0, 5.0))
            recommendations.append((movie_id, clipped_score))

        return recommendations

    def _cold_start_recommendations(self, n: int) -> list[tuple[int, float]]:
        """Return globally top rated items for cold-start users."""
        if self.ratings_sparse is None:
            return []
        item_sums = np.array(self.ratings_sparse.sum(axis=0)).flatten()
        item_counts = np.diff(self.ratings_sparse.tocsc().indptr)
        with np.errstate(divide="ignore", invalid="ignore"):
            item_means = np.where(item_counts > 5, item_sums / item_counts, 0.0)
        top_indices = np.argsort(item_means)[::-1][:n]
        return [(self.idx_to_item[idx], float(item_means[idx])) for idx in top_indices if idx in self.idx_to_item]

    def save(self, filepath: str | Path) -> None:
        """Serialize fitted model to disk using joblib.

        Args:
            filepath: Destination file path (e.g. .pkl or .joblib).
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved MatrixFactorizationALS to %s", path)

    @classmethod
    def load(cls, filepath: str | Path) -> MatrixFactorizationALS:
        """Load serialized model from disk.

        Args:
            filepath: Source file path.

        Returns:
            Deserialized MatrixFactorizationALS instance.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"ALS model file not found: {path}")
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is of type {type(model)}, expected {cls}")
        logger.info("Loaded MatrixFactorizationALS from %s", path)
        return model


class NeuralCollaborativeFilter:
    """Neural Collaborative Filtering (NCF / NeuMF) recommender.

    Combines Generalized Matrix Factorization (GMF) and Deep Multi-Layer
    Perceptron (MLP) branches to model both linear interactions and non-linear
    feature interactions through dual user/item embeddings.
    """

    def __init__(
        self,
        latent_dim_gmf: int = 32,
        latent_dim_mlp: int = 32,
        mlp_hidden_units: tuple[int, ...] = (128, 64, 32),
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        random_state: int = 42,
    ) -> None:
        """Initialize NeuMF architecture hyperparameters.

        Args:
            latent_dim_gmf: Embedding dimension for GMF stream.
            latent_dim_mlp: Embedding dimension for MLP stream.
            mlp_hidden_units: Hidden layer neuron counts for MLP stream.
            dropout_rate: Dropout regularization probability.
            learning_rate: Adam optimizer learning rate.
            random_state: Random initialization seed.
        """
        self.latent_dim_gmf = latent_dim_gmf
        self.latent_dim_mlp = latent_dim_mlp
        self.mlp_hidden_units = mlp_hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model: Any = None
        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}
        self.global_mean: float = 3.5

    def _build_model(self, n_users: int, n_items: int) -> Any:
        """Construct the Keras NeuMF model graph."""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models, regularizers
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for NeuralCollaborativeFilter. "
                "Install it via `pip install tensorflow`."
            ) from exc

        tf.random.set_seed(self.random_state)

        user_input = layers.Input(shape=(1,), name="user_id", dtype="int32")
        item_input = layers.Input(shape=(1,), name="item_id", dtype="int32")

        # --- GMF Branch ---
        gmf_user_emb = layers.Embedding(
            input_dim=n_users,
            output_dim=self.latent_dim_gmf,
            embeddings_initializer="he_normal",
            embeddings_regularizer=regularizers.l2(1e-6),
            name="gmf_user_emb",
        )(user_input)
        gmf_item_emb = layers.Embedding(
            input_dim=n_items,
            output_dim=self.latent_dim_gmf,
            embeddings_initializer="he_normal",
            embeddings_regularizer=regularizers.l2(1e-6),
            name="gmf_item_emb",
        )(item_input)

        gmf_user_flat = layers.Flatten()(gmf_user_emb)
        gmf_item_flat = layers.Flatten()(gmf_item_emb)
        gmf_vector = layers.Multiply(name="gmf_product")([gmf_user_flat, gmf_item_flat])

        # --- MLP Branch ---
        mlp_user_emb = layers.Embedding(
            input_dim=n_users,
            output_dim=self.latent_dim_mlp,
            embeddings_initializer="he_normal",
            embeddings_regularizer=regularizers.l2(1e-6),
            name="mlp_user_emb",
        )(user_input)
        mlp_item_emb = layers.Embedding(
            input_dim=n_items,
            output_dim=self.latent_dim_mlp,
            embeddings_initializer="he_normal",
            embeddings_regularizer=regularizers.l2(1e-6),
            name="mlp_item_emb",
        )(item_input)

        mlp_user_flat = layers.Flatten()(mlp_user_emb)
        mlp_item_flat = layers.Flatten()(mlp_item_emb)
        mlp_vector = layers.Concatenate(name="mlp_concat")([mlp_user_flat, mlp_item_flat])

        for idx, units in enumerate(self.mlp_hidden_units):
            mlp_vector = layers.Dense(units, activation="relu", name=f"mlp_dense_{idx}")(mlp_vector)
            mlp_vector = layers.BatchNormalization(name=f"mlp_bn_{idx}")(mlp_vector)
            mlp_vector = layers.Dropout(self.dropout_rate, name=f"mlp_drop_{idx}")(mlp_vector)

        # --- NeuMF Output Fusion ---
        fusion = layers.Concatenate(name="neumf_fusion")([gmf_vector, mlp_vector])
        prediction = layers.Dense(1, activation="linear", name="rating_output")(fusion)

        keras_model = models.Model(inputs=[user_input, item_input], outputs=prediction, name="NeuMF")
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        keras_model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae", "mse"])
        return keras_model

    def fit(
        self,
        ratings_df: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating",
        epochs: int = 15,
        batch_size: int = 256,
        validation_split: float = 0.1,
        verbose: int = 1,
    ) -> NeuralCollaborativeFilter:
        """Train NeuMF model using user-item ratings DataFrame.

        Args:
            ratings_df: Interaction dataset containing user, item, and rating columns.
            user_col: Column name for user ID.
            item_col: Column name for item ID.
            rating_col: Column name for numeric rating.
            epochs: Maximum training epochs.
            batch_size: Mini-batch gradient descent size.
            validation_split: Fraction of ratings used for validation monitoring.
            verbose: Verbosity level for training logs (0, 1, or 2).

        Returns:
            Fitted NeuralCollaborativeFilter instance.
        """
        unique_users = sorted(ratings_df[user_col].unique())
        unique_items = sorted(ratings_df[item_col].unique())

        self.user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx_to_user = {i: uid for uid, i in self.user_to_idx.items()}
        self.item_to_idx = {iid: j for j, iid in enumerate(unique_items)}
        self.idx_to_item = {j: iid for iid, j in self.item_to_idx.items()}
        self.global_mean = float(ratings_df[rating_col].mean())

        u_indices = ratings_df[user_col].map(self.user_to_idx).to_numpy(dtype=np.int32)
        i_indices = ratings_df[item_col].map(self.item_to_idx).to_numpy(dtype=np.int32)
        y_ratings = ratings_df[rating_col].to_numpy(dtype=np.float32)

        n_users = len(unique_users)
        n_items = len(unique_items)
        logger.info(
            "Building NeuMF model for %d users and %d items...",
            n_users,
            n_items,
        )
        self.model = self._build_model(n_users=n_users, n_items=n_items)

        try:
            from tensorflow.keras.callbacks import EarlyStopping

            callbacks = [
                EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
            ]
        except ImportError:
            callbacks = []

        logger.info("Training NeuMF: epochs=%d, batch_size=%d", epochs, batch_size)
        self.model.fit(
            x=[u_indices, i_indices],
            y=y_ratings,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
        )
        logger.info("NeuMF training successfully concluded.")
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a given user and movie.

        Args:
            user_id: User identifier.
            movie_id: Item identifier.

        Returns:
            Predicted rating bounded between 1.0 and 5.0.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        u_idx = self.user_to_idx.get(user_id)
        i_idx = self.item_to_idx.get(movie_id)

        if u_idx is None or i_idx is None:
            return self.global_mean

        pred = self.model.predict(
            [np.array([u_idx], dtype=np.int32), np.array([i_idx], dtype=np.int32)],
            verbose=0,
        )
        score = float(pred[0][0])
        return float(np.clip(score, 1.0, 5.0))

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        all_movie_ids: list[int] | None = None,
        user_rated_movies: set[int] | None = None,
        batch_size: int = 1024,
    ) -> list[tuple[int, float]]:
        """Generate top-N recommendations for user.

        Args:
            user_id: User identifier.
            n: Number of movies to return.
            all_movie_ids: Optional list of movie IDs to evaluate. If None, uses all known items.
            user_rated_movies: Set of movie IDs to exclude because user has already seen them.
            batch_size: Prediction batch size.

        Returns:
            List of (movie_id, predicted_score) tuples sorted in descending order.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        u_idx = self.user_to_idx.get(user_id)
        if u_idx is None:
            logger.warning("Unknown user %d for NeuMF, falling back to empty/cold-start", user_id)
            return []

        target_items = all_movie_ids if all_movie_ids is not None else list(self.item_to_idx.keys())
        if user_rated_movies:
            target_items = [m for m in target_items if m not in user_rated_movies]

        candidate_i_indices = [self.item_to_idx[m] for m in target_items if m in self.item_to_idx]
        candidate_movie_ids = [m for m in target_items if m in self.item_to_idx]

        if not candidate_i_indices:
            return []

        user_arr = np.full(len(candidate_i_indices), fill_value=u_idx, dtype=np.int32)
        item_arr = np.array(candidate_i_indices, dtype=np.int32)

        preds = self.model.predict([user_arr, item_arr], batch_size=batch_size, verbose=0).flatten()
        top_k_indices = np.argsort(preds)[::-1][:n]

        return [
            (candidate_movie_ids[idx], float(np.clip(preds[idx], 1.0, 5.0)))
            for idx in top_k_indices
        ]

    def save(self, directory_path: str | Path) -> None:
        """Save Keras model weights and metadata mappings.

        Args:
            directory_path: Directory path where model and mappings will be saved.
        """
        target_dir = Path(directory_path)
        target_dir.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            model_file = target_dir / "ncf_model.keras"
            self.model.save(model_file)
            logger.info("Saved Keras model to %s", model_file)

        metadata = {
            "latent_dim_gmf": self.latent_dim_gmf,
            "latent_dim_mlp": self.latent_dim_mlp,
            "mlp_hidden_units": self.mlp_hidden_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "global_mean": self.global_mean,
            "user_to_idx": self.user_to_idx,
            "idx_to_user": self.idx_to_user,
            "item_to_idx": self.item_to_idx,
            "idx_to_item": self.idx_to_item,
        }
        metadata_file = target_dir / "ncf_metadata.joblib"
        joblib.dump(metadata, metadata_file)
        logger.info("Saved NCF metadata to %s", metadata_file)

    @classmethod
    def load(cls, directory_path: str | Path) -> NeuralCollaborativeFilter:
        """Load saved NeuMF model and metadata.

        Args:
            directory_path: Path containing model artifacts.

        Returns:
            Restored NeuralCollaborativeFilter instance.
        """
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("TensorFlow required to load NeuralCollaborativeFilter.") from exc

        target_dir = Path(directory_path)
        metadata_file = target_dir / "ncf_metadata.joblib"
        model_file = target_dir / "ncf_model.keras"

        if not metadata_file.is_file() or not model_file.is_file():
            raise FileNotFoundError(f"Missing required NCF artifacts in {target_dir}")

        metadata: dict[str, Any] = joblib.load(metadata_file)
        instance = cls(
            latent_dim_gmf=metadata["latent_dim_gmf"],
            latent_dim_mlp=metadata["latent_dim_mlp"],
            mlp_hidden_units=metadata["mlp_hidden_units"],
            dropout_rate=metadata["dropout_rate"],
            learning_rate=metadata["learning_rate"],
        )
        instance.user_to_idx = metadata["user_to_idx"]
        instance.idx_to_user = metadata["idx_to_user"]
        instance.item_to_idx = metadata["item_to_idx"]
        instance.idx_to_item = metadata["idx_to_item"]
        instance.global_mean = metadata["global_mean"]

        instance.model = tf.keras.models.load_model(model_file)
        logger.info("Loaded NeuralCollaborativeFilter successfully from %s", target_dir)
        return instance
