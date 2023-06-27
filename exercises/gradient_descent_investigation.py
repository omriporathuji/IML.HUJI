import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from sklearn.metrics import roc_curve
from IMLearn.model_selection.cross_validate import cross_validate
from IMLearn.metrics.loss_functions import misclassification_error

import plotly.graph_objects as go


LAMBDAS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
MAX_ITER = 20000
LR = 1e-4

def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights_list = [], []

    def callback(weights, val, **kwargs):
        values.append(val)
        weights_list.append(weights)

    return callback, values, weights_list


def plot_convergence_rate(values, title):
    return go.Figure([go.Scatter(x=list(range(1, len(values) + 1)), y=values, mode="markers+lines")],
                     layout=go.Layout(title=title, xaxis_title="Iteration", yaxis_title="Norm"))


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    min_norms, min_etas = [np.inf, np.inf], [0, 0]
    indices = {'L1': 0, 'L2': 1}
    for eta in etas:
        for name, module in [('L1', L1), ('L2', L2)]:
            c, values, weights_list = get_gd_state_recorder_callback()
            gd = GradientDescent(learning_rate=FixedLR(eta), callback=c)
            gd.fit(module(np.array(init)), None, None)
            fig = plot_descent_path(module, np.array(weights_list),
                                    title="Descent Path for {} module, eta = {}".format(name, eta))
            fig.update_layout(width=1000, height=1000)
            # fig.show()
            values = np.array(values)
            if name == 'L2':
                values = np.sqrt(np.array(values))

            if values[-1] < min_norms[indices[name]]:
                min_norms[indices[name]] = values[-1]
                min_etas[indices[name]] = eta

            # plot_convergence_rate(values, "Convergence Rate for {} Module, eta = {}".format(name, eta)). \
            #     update_layout(width=1000, height=1000).show()

    print('Minimal loss for L1: {}, achieved with eta = {}'.format(min_norms[indices['L1']],
                                                                   min_etas[indices['L1']]))
    print('Minimal loss for L2: {}, achieved with eta = {}'.format(min_norms[indices['L2']],
                                                                   min_etas[indices['L2']]))


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    lr = LogisticRegression()
    lr.fit(np.array(X_train), np.array(y_train))
    y_prob = lr.predict_proba(np.array(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title="ROC Curve Of Fitted Model",
                         xaxis=dict(title="<b>False Positive Rate (FPR)</b>"),
                         yaxis=dict(title="<b>True Positive Rate (TPR)</b>")))
    fig.show()

    best_alpha = thresholds[np.argmax(tpr - fpr)]
    lr.alpha_ = best_alpha
    loss = lr.loss(np.array(X_test), np.array(y_test))

    print("Best alpha is {}. Model loss with said alpha is {}".format(best_alpha, loss))

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    for pen in ['l1', 'l2']:
        scores = []
        solver = GradientDescent(max_iter=MAX_ITER, learning_rate=FixedLR(base_lr=LR))

        for lam in LAMBDAS:
            reg_lr = LogisticRegression(penalty=pen, lam=lam, solver=solver)
            scores.append(cross_validate(reg_lr, X_train.to_numpy(), y_train.to_numpy(), misclassification_error)[1])

        scores = np.array(scores)
        best_lam_index = np.argmin(scores)
        best_lam = LAMBDAS[best_lam_index]
        reg_lr = LogisticRegression(penalty=pen, lam=best_lam, solver=solver)
        reg_lr.fit(X_train.to_numpy(), y_train.to_numpy())
        print(f"Best lambda for {pen} penalty is {best_lam}. Test error is {reg_lr.loss(X_test.to_numpy(), y_test.to_numpy())}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
