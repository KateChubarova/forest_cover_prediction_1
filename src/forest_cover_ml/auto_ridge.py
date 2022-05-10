from pathlib import Path

from joblib import dump

import click
import mlflow
import mlflow.sklearn

from src.forest_cover_ml.data import get_dataset
from sklearn.linear_model import Ridge

from sklearn.model_selection import KFold, RandomizedSearchCV


@click.command()
@click.option(
    "--dataset-path-train",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--splits",
    default='4',
    type=int,
    show_default=True,
)
@click.option(
    "--feature-selection",
    default=False,
    type=bool,
    show_default=True,
)
def train(
        dataset_path_train: Path,
        save_model_path: Path,
        random_state: int,
        test_split_ratio: float,
        splits: int,
        feature_selection: bool,

):
    with mlflow.start_run():
        features_train, features_val, target_train, target_val = get_dataset(
            dataset_path_train, test_split_ratio, random_state, feature_selection
        )

        rr = Ridge()
        kfold = KFold(n_splits=splits, shuffle=True, random_state=random_state)

        param_dist = {"max_iter": [1000, 1500, 2000, 5000],
                      "solver": ["svd", "cholesky", "saga"],
                      "tol": [0.1, 0.01, 0.001, 0.0001]}

        random_search = RandomizedSearchCV(
            estimator=rr,
            param_distributions=param_dist,
            n_iter=10,
            cv=kfold)

        random_search.fit(features_train, target_train)

        mlflow.log_param("n_splits", splits)
        mlflow.log_param("classifier", "ridge regression")
        mlflow.log_param("max_iter", random_search.best_params_['max_iter'])
        mlflow.log_param("solver", random_search.best_params_['solver'])
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("tol", random_search.best_params_['tol'])
        click.echo("Model: random forest regression")
        mlflow.log_param("accuracy", random_search.best_score_)

        dump(random_search, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
