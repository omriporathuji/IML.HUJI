import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ab = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)

    train_losses = [ab.partial_loss(train_X, train_y, n) for n in range(1, n_learners + 1)]
    test_losses = [ab.partial_loss(test_X, test_y, n) for n in range(1, n_learners + 1)]
    iterations = list(range(1, n_learners + 1))

    fig1 = go.Figure(data=[go.Scatter(x=iterations, y=train_losses, name="Train Error", mode='lines'),
                           go.Scatter(x=iterations, y=test_losses, name="Test Error", mode='lines')]).update_layout(
        title_text=f"<b>Adaboost Train and Test Errors by Number of Iterations, Noise={noise}</b>", title_font_size=25,
        margin=dict(t=75)
    ).update_xaxes(
        title_text='Iteration', title_font_size=20).update_yaxes(title_text='Misclassification Error',
                                                                 title_font_size=20)
    fig1.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig2 = make_subplots(rows=1, cols=len(T), subplot_titles=[f"{num} Classifiers" for num in T],
                         horizontal_spacing=0.01).update_layout(
        title_text=f"<b>Decision Boundaries by Number of Classifiers, Noise={noise}</b>")
    for i, t in enumerate(T):
        fig2.add_traces(
            [decision_surface(lambda X: ab.partial_predict(X, t), lims[0], lims[1], showscale=False, density=60),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers', showlegend=False,
                        marker=dict(color=test_y, symbol=np.where(test_y == 1, 'circle', 'x'), size=12))],
            rows=1, cols=i + 1).update_layout(margin=dict(t=75), height=500, width=2000,
                                              title_font_size=25).update_xaxes(visible=False).update_yaxes(
            visible=False)
    fig2.show()

    # Question 3: Decision surface of best performing ensemble
    test_losses = np.array(test_losses)
    best_num = np.argmin(test_losses) + 1
    fig3 = go.Figure([decision_surface(lambda X: ab.partial_predict(X, best_num), lims[0], lims[1], density=60,
                                       showscale=False),
                      go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers', showlegend=False,
                                 marker=dict(color=test_y, symbol=np.where(test_y == 1, 'circle', 'x'),
                                             size=12))]).update_layout(
        title_text=f"<b>Best Ensemble Size: {best_num}, Accuracy: {1 - np.round(test_losses[best_num - 1], 2)} <br /> Noise={noise}</b>",
        title_font_size=25, margin=dict(t=75), height=750, width=750).update_xaxes(visible=False).update_yaxes(
        visible=False)
    fig3.show()

    # Question 4: Decision surface with weighted samples
    D = 20 * ab.D_ / np.max(ab.D_)
    fig4 = go.Figure([decision_surface(ab.predict, lims[0], lims[1], density=60,
                                       showscale=False),
                      go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers', showlegend=False,
                                 marker=dict(color=train_y, symbol=np.where(train_y == 1, 'circle', 'x'),
                                             size=D))]).update_layout(
        title_text=f"<b>AdaBoost Distribution of Final Ensemble <br />Noise={noise}</b>",
        title_font_size=25, margin=dict(t=75), height=750, width=750).update_xaxes(visible=False).update_yaxes(
        visible=False)
    fig4.show()


if __name__ == '__main__':
    import time

    start_time = time.time()

    np.random.seed(0)
    for noise in [0, 0.4]:
        fit_and_evaluate_adaboost(noise=noise)
    print("--- %s seconds ---" % (time.time() - start_time))
