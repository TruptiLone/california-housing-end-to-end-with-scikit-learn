"""
California Housing, End to End with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_housing
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

def load_housing():
    tarball = Path(tempfile.gettempdir()) / "housing.tgz"
    if not tarball.exists():
        urlretrieve(
            "https://github.com/ageron/data/raw/main/housing.tgz",
            tarball,
        )
    with tarfile.open(tarball, "r:gz") as archive:
        with archive.extractfile("housing/housing.csv") as csv_file:
            return pd.read_csv(csv_file)

# Step 2 - income_categories
import numpy as np
import pandas as pd

def income_categories(df):
    # Bin median income into five categories, preserving the input index.
    return pd.cut(
        df["median_income"],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

# Step 3 - stratified_split
from sklearn.model_selection import train_test_split
def stratified_split(df, test_size=0.2, random_state=42):
    train_set, test_set = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=income_categories(df),
    )
    return train_set, test_set

# Step 4 - explore_correlations
def explore_correlations(df):
    # Compute Pearson correlations using only numeric columns.
    correlations = df.corr(method="pearson", numeric_only=True)

    # Exclude the target itself and rank from most positive to most negative.
    return (
        correlations["median_house_value"]
        .drop("median_house_value")
        .sort_values(ascending=False)
    )

# Step 5 - add_ratio_features
def add_ratio_features(df):
    result = df.copy()

    result["rooms_per_house"] = result["total_rooms"] / result["households"]
    result["bedrooms_ratio"] = result["total_bedrooms"] / result["total_rooms"]
    result["people_per_house"] = result["population"] / result["households"]

    return result

# Step 6 - split_features_labels
def split_features_labels(df):
    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"].copy()
    return X, y

# Step 7 - ClusterSimilarity
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        # Store parameters for scikit-learn compatibility.
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state,
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(
            X, self.kmeans_.cluster_centers_, gamma=self.gamma
        )

    def get_feature_names_out(self, names=None):
        return [
            f"Cluster {i} similarity"
            for i in range(self.n_clusters)
        ]

# Step 8 - numeric_pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def numeric_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )

# Step 9 - categorical_pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

def normalize_missing(X):
    """Treat both Python None and CSV NaN as missing categorical values."""
    return pd.DataFrame(X).where(pd.notna(X), np.nan)


def categorical_pipeline():
    return make_pipeline(
        FunctionTransformer(normalize_missing, feature_names_out="one-to-one"),
        SimpleImputer(
            strategy="most_frequent",
            ),
        OneHotEncoder(handle_unknown="ignore"),
    )

# Step 10 - build_preprocessing
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

def build_preprocessing(n_clusters=10, gamma=1.0, random_state=42):
    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler(),
    )

    return ColumnTransformer(
        transformers=[
            (
                "log",
                log_pipeline,
                [
                    "total_bedrooms",
                    "total_rooms",
                    "population",
                    "households",
                    "median_income",
                ],
            ),
            (
                "geo",
                ClusterSimilarity(
                    n_clusters=n_clusters,
                    gamma=gamma,
                    random_state=random_state,
                ),
                ["latitude", "longitude"],
            ),
            (
                "cat",
                categorical_pipeline(),
                ["ocean_proximity"],
            ),
        ],
        remainder=numeric_pipeline(),
    )

# Step 11 - rmse
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# Step 12 - dummy_baseline_rmse
from sklearn.dummy import DummyRegressor
def dummy_baseline_rmse(X, y):
    model = DummyRegressor(strategy="mean")
    model.fit(X, y)

    return rmse(y, model.predict(X))

# Step 13 - cross_val_rmse
from sklearn.model_selection import cross_val_score
def cross_val_rmse(model, X, y, cv=3):
    # Negate scikit-learn's negative RMSE scores to get positive errors.
    scores = -cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
    )

    return {
        "scores": [float(score) for score in scores],
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }

# Step 14 - linear_model
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def linear_model(preprocessing):
    # Combine preprocessing and regression without fitting either step.
    return make_pipeline(preprocessing, LinearRegression())

# Step 15 - forest_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

def forest_model(preprocessing, n_estimators=50, random_state=42):
    return make_pipeline(
        preprocessing,
        RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
        ),
    )

# Step 16 - random_search
from sklearn.model_selection import RandomizedSearchCV

def random_search(pipeline, X, y, n_iter=5, cv=3, random_state=42):
    param_distributions = {
        "columntransformer__geo__n_clusters": range(3, 11),
        "randomforestregressor__max_features": range(2, 9),
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
    )
    search.fit(X, y)
    return search

# Step 17 - test_rmse
def test_rmse(model, test_set):
    # Prepare test inputs without modifying the original test set.
    test_with_ratios = add_ratio_features(test_set)
    X_test, y_test = split_features_labels(test_with_ratios)

    # Use the fitted pipeline without retraining.
    predictions = model.predict(X_test)
    return rmse(y_test, predictions)

# Step 18 - bootstrap_rmse_ci
def bootstrap_rmse_ci(y_true, y_pred, n_boot=200, alpha=0.05, random_state=42):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(random_state)

    # Resample matching true/predicted pairs to preserve their alignment.
    bootstrap_rmses = []
    for _ in range(n_boot):
        indices = rng.integers(0, n, n)
        bootstrap_rmses.append(rmse(y_true[indices], y_pred[indices]))

    low, high = np.percentile(
        bootstrap_rmses,
        [100 * alpha / 2, 100 * (1 - alpha / 2)],
    )
    return float(low), float(high)

# Step 19 - feature_importances
def feature_importances(search, k=5):
    pipeline = search.best_estimator_
    names = pipeline.steps[0][1].get_feature_names_out()
    importances = pipeline.steps[-1][1].feature_importances_

    # Rank using full precision before rounding the returned values.
    ranked = sorted(
        zip(importances, names),
        key=lambda pair: pair[0],
        reverse=True,
    )

    return [
        (round(float(importance), 3), name)
        for importance, name in ranked[:k]
    ]

# Step 20 - worst_errors
import pandas as pd

def worst_errors(model, df, k=5):
    X, y = split_features_labels(add_ratio_features(df))
    predictions = model.predict(X)

    errors = pd.DataFrame(
        {"actual": y, "predicted": predictions},
        index=df.index,
    )
    errors["abs_error"] = (errors["actual"] - errors["predicted"]).abs()

    return errors.sort_values("abs_error", ascending=False).head(k)

# Step 21 - save_and_reload
import joblib

def save_and_reload(model, path):
    joblib.dump(model, path)
    return joblib.load(path)

# Step 22 - predict_new
def predict_new(model, districts):
    df = pd.DataFrame(districts)
    X = add_ratio_features(df)
    predictions = model.predict(X)

    return [float(round(value)) for value in predictions]

