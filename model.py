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
    # TODO: Download housing.tgz once into tempfile.gettempdir() and read housing/housing.csv from it.
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
    # TODO: pd.cut median_income with edges [0, 1.5, 3, 4.5, 6, inf] and labels 1..5; return an int Series.
       # Bin median income into five categories, preserving the input index.
    return pd.cut(
        df["median_income"],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

# Step 3 - stratified_split
from sklearn.model_selection import train_test_split
def stratified_split(df, test_size=0.2, random_state=42):
    # TODO: train_test_split stratified on income_categories(df); return (train_set, test_set).
    train_set, test_set = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=income_categories(df),
    )
    return train_set, test_set

# Step 4 - explore_correlations
def explore_correlations(df):
    # TODO: Pearson correlation of every numeric column with median_house_value, sorted descending, target excluded.
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
    # TODO: Return a copy with rooms_per_house, bedrooms_ratio and people_per_house columns added.
    result = df.copy()

    result["rooms_per_house"] = result["total_rooms"] / result["households"]
    result["bedrooms_ratio"] = result["total_bedrooms"] / result["total_rooms"]
    result["people_per_house"] = result["population"] / result["households"]

    return result

# Step 6 - split_features_labels
def split_features_labels(df):
    # TODO: Return (X without median_house_value, y = median_house_value Series).
    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"].copy()
    return X, y

# Step 7 - ClusterSimilarity (not yet solved)
# TODO: implement

# Step 8 - numeric_pipeline (not yet solved)
# TODO: implement

# Step 9 - categorical_pipeline (not yet solved)
# TODO: implement

# Step 10 - build_preprocessing (not yet solved)
# TODO: implement

# Step 11 - rmse (not yet solved)
# TODO: implement

# Step 12 - dummy_baseline_rmse (not yet solved)
# TODO: implement

# Step 13 - cross_val_rmse (not yet solved)
# TODO: implement

# Step 14 - linear_model (not yet solved)
# TODO: implement

# Step 15 - forest_model (not yet solved)
# TODO: implement

# Step 16 - random_search (not yet solved)
# TODO: implement

# Step 17 - test_rmse (not yet solved)
# TODO: implement

# Step 18 - bootstrap_rmse_ci (not yet solved)
# TODO: implement

# Step 19 - feature_importances (not yet solved)
# TODO: implement

# Step 20 - worst_errors (not yet solved)
# TODO: implement

# Step 21 - save_and_reload (not yet solved)
# TODO: implement

# Step 22 - predict_new (not yet solved)
# TODO: implement

