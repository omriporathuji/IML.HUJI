from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    indices = np.arange(X.shape[0])
    folds = np.array_split(indices, cv)
    train_score = 0
    validation_score = 0

    for f in folds:
        X_train = np.delete(X, f, axis=0)
        y_train = np.delete(y, f, axis=0)
        X_validation = X[f]
        y_validation = y[f]
        e = estimator.fit(X_train, y_train)
        train_score += scoring(y_train, e.predict(X_train))
        validation_score += scoring(y_validation, e.predict(X_validation))

    return train_score / cv, validation_score / cv