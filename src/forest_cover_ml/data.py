from pathlib import Path
from typing import Tuple

import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE


def get_dataset(
        csv_path_train: Path, test_split_ratio: float, random_state: int, feature_selection: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path_train)
    click.echo(f"Train dataset shape: {dataset.shape}.")

    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]

    if feature_selection:
        features = TSNE(n_components=3).fit_transform(features)

    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_split_ratio, random_state=random_state
    )
    return features_train, features_val, target_train, target_val
