from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FOLDS = 5

def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)

    X_train, y_train, X_test, y_test = split_train_test(X, y, train_proportion=(50 / len(y)))
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    ridge_range = np.linspace(.0001, 2, num=n_evaluations)
    lasso_range = np.linspace(.001, 4, num=n_evaluations)

    ridge_scores, lasso_scores = [], []
    for r, l in zip(ridge_range, lasso_range):
        ridge_scores.append(cross_validate(RidgeRegression(r), X_train, y_train, scoring=mean_square_error, cv=FOLDS))
        lasso_scores.append(cross_validate(Lasso(l, max_iter=5000), X_train, y_train, scoring=mean_square_error, cv=FOLDS))

    ridge_scores = np.array(ridge_scores)
    lasso_scores = np.array(lasso_scores)

    fig = make_subplots(1, 2, subplot_titles=["Ridge", "Lasso"])\
        .update_layout(title_text=f"<b>Train and Validation Errors over {FOLDS} folds </b>", title_font_size=28,
                       margin=dict(t=75))\
        .update_xaxes(title_text=r"$Regularization Parameter (\lambda\text)$", title_font_size=20)\
        .update_yaxes(title_text="MSE")\
        .add_traces([go.Scatter(x=ridge_range, y=ridge_scores[:, 0], name="Ridge Train Error"),
                     go.Scatter(x=ridge_range, y=ridge_scores[:, 1], name="Ridge Validation Error"),
                     go.Scatter(x=lasso_range, y=lasso_scores[:, 0], name="Lasso Train Error"),
                     go.Scatter(x=lasso_range, y=lasso_scores[:, 1], name="Lasso Validation Error")],
                    rows=[1, 1, 1, 1], cols=[1, 1, 2, 2])
    fig.show()

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    chosen_ridge = ridge_range[np.argmin(ridge_scores[:, 1])]
    chosen_lasso = lasso_range[np.argmin(lasso_scores[:, 1])]

    print("Chosen Ridge:", chosen_ridge)
    print("Chosen Lasso:", chosen_lasso)

    print("LS Error:", LinearRegression().fit(X_train, y_train).loss(X_test, y_test))
    print("Ridge Error:", RidgeRegression(lam=chosen_ridge).fit(X_train, y_train).loss(X_test, y_test))
    lasso_prediction = Lasso(alpha=chosen_lasso).fit(X_train, y_train).predict(X_test)
    print("Lasso Error:", mean_square_error(y_test, lasso_prediction))


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
