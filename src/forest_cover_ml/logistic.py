from pathlib import Path

from joblib import dump

import click
import mlflow
import mlflow.sklearn

from src.forest_cover_ml.data import get_dataset
from src.forest_cover_ml.pipeline import create_pipeline_lr

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score

import numpy as np


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
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=5000,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=2.0,
    type=float,
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
@click.option(
    "--tol",
    default=0.0001,
    type=float,
    show_default=True,
)
@click.option(
    "--penalty",
    default='none',
    type=str,
    show_default=True,
)

def train(
        dataset_path_train: Path,
        save_model_path: Path,
        use_scaler: bool,
        max_iter: int,
        logreg_c: float,
        random_state: int,
        test_split_ratio: float,
        splits: int,
        feature_selection: bool,
        penalty: str,
        tol: float
) -> None:
    with mlflow.start_run():
        features_train, features_val, target_train, target_val = get_dataset(
            dataset_path_train, test_split_ratio, random_state, feature_selection
        )

        pipeline = create_pipeline_lr(use_scaler, max_iter, logreg_c, random_state, tol, penalty)

        log_params(mlflow, penalty, max_iter, logreg_c, use_scaler, tol, feature_selection, random_state)
        click.echo("Model: logistic regressor")

        pipeline.fit(features_train, target_train)

        kfold = KFold(n_splits=splits, shuffle=True, random_state=random_state)

        mae = make_scorer(mean_absolute_error)
        mse = make_scorer(mean_squared_error)

        scores_mae = cross_val_score(pipeline, features_train, target_train, cv=kfold, scoring=mae)
        scores_mse = cross_val_score(pipeline, features_train, target_train, cv=kfold, scoring=mse)
        scores_accuracy = cross_val_score(pipeline, features_train, target_train, cv=kfold)

        log_metrics(scores_accuracy, scores_mse, scores_mae)

        mlflow.log_param("n_splits", splits)

        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")


def log_metrics(scores_accuracy, scores_mse, scores_mae):
    mlflow.log_metric("accuracy", np.mean(scores_accuracy))
    mlflow.log_metric("mse", np.mean(scores_mse))
    mlflow.log_metric("mae", np.mean(scores_mae))


def log_params(mlflow, penalty, max_iter, logreg_c, use_scaler, tol, feature_selection, random_state):
    mlflow.log_param("classifier", 'logistic regression')
    mlflow.log_param("penalty", penalty)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("logreg_c", logreg_c)
    mlflow.log_param("use_scaler", use_scaler)
    mlflow.log_param("tol", tol)
    mlflow.log_param("feature_selection", feature_selection)
    mlflow.log_param("random_state", random_state)
