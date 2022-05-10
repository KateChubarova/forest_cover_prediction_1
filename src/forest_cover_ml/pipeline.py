from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_pipeline_lr(use_scaler: bool, max_iter: int, logreg_C: float,
                       random_state: int, tol: float, penalty: str) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(random_state=random_state, max_iter=max_iter, C=logreg_C, penalty=penalty, tol=tol),
        )
    )

    return Pipeline(steps=pipeline_steps)


def create_pipeline_rr(use_scaler: bool, max_iter: int, random_state: int, tol: float,
                       solver: str) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", MinMaxScaler()))

    pipeline_steps.append(
        (
            "classifier",
            Ridge(random_state=random_state, max_iter=max_iter, tol=tol,
                  solver=solver),
        )
    )
    return Pipeline(steps=pipeline_steps)
