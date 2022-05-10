Model evaluation and selection for forest cover.

This demo uses [Heart Disease](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) dataset.

## Usage
This package allows you to train model for detecting the presence of heart disease in the patient.
1. Clone this repository to your machine.
2. Download [Heart Disease](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) dataset, save csv locally (default path is *data/heart.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.11).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run train with the following command:
Logistic regression:
```sh
poetry run train_logistic -d <path to csv with data> -s <path to save trained model>
```
Ridge regression:
```sh
poetry run train_ridge -d <path to csv with data> -s <path to save trained model>
```
Auto logistic regression:
```sh
poetry run auto_logistic -d <path to csv with data> -s <path to save trained model>
```
Auto ridge regression:
```sh
poetry run auto_logistic -d <path to csv with data> -s <path to save trained model>
```

You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```

## Development

The code in this repository must be tested, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer instruments, e.g. pytest:
```
poetry run pytest
```
More conveniently, to run all sessions of testing and formatting in a single command, install and use [nox](https://nox.thea.codes/en/stable/): 
```
nox [-r]
```
Format your code with [black](https://github.com/psf/black) by using either nox or poetry:
```
nox -[r]s black
poetry run black src tests noxfile.py
