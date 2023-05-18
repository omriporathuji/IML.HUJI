from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import os

DATASETS_PATH = '../datasets'
FIGS_DIR = 'figs'


def save_figure(fig, figure_name, fig_dir):
    """
    Save a figure in the given folder, and create the folder if it doesn't exist
    """
    try:
        if not os.path.exists(os.path.join(os.getcwd(), fig_dir)):
            os.mkdir(os.path.join(os.getcwd(), fig_dir))
        fig.write_html(os.path.join(os.getcwd(), fig_dir, figure_name))
    except:
        print("Could not save figure")


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(os.path.join(DATASETS_PATH, f))

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        callback = lambda P, _, __: losses.append(P.loss(X, y))
        perceptron = Perceptron(max_iter=1000, callback=callback)
        perceptron.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = px.line(x=range(len(losses)), y=losses, labels={'x': 'Iteration', 'y': 'Misclassification Error'},
                      markers=True, title="<b>Perceptron Training Loss by Iteration on {} data</b>".format(n))
        fig.update_layout(title_font_size=30, margin=dict(t=75))
        fig.update_xaxes(title_font_size=20)
        fig.update_yaxes(title_font_size=20)

        save_figure(fig, 'perceptron_{}.html'.format(f.split('.')[0]), FIGS_DIR)


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(os.path.join(DATASETS_PATH, f))
        bayes = GaussianNaiveBayes()
        lda = LDA()

        # Fit models and predict over training set
        bayes.fit(X, y)
        bayes_y = bayes.predict(X)

        lda.fit(X, y)
        lda_y = lda.predict(X)

        # Plot a figure with two subplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots

        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            f"Gaussian Naive Bayes: Accuracy = {np.round(accuracy(y, bayes_y), 4) * 100}%",
            f"LDA: Accuracy = {np.round(accuracy(y, lda_y), 4) * 100}%"])

        # Add traces for data-points setting symbols and colors
        fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker={'color': bayes_y,
                                                                                 'symbol': class_symbols[y],
                                                                                 'colorscale': class_colors(3),
                                                                                 'size': 11}),
                        go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker={'color': lda_y,
                                                                                 'symbol': class_symbols[y],
                                                                                 'colorscale': class_colors(3),
                                                                                 'size': 11})],
                       rows=[1, 1], cols=[1, 2])

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_traces([go.Scatter(x=bayes.mu_[:, 0], y=bayes.mu_[:, 1], mode='markers', marker={'symbol': 'x',
                                                                                                 'color': 'black',
                                                                                                 'size': 15}),
                        go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode='markers', marker={'symbol': 'x',
                                                                                             'color': 'black',
                                                                                             'size': 15})],
                       rows=[1, 1], cols=[1, 2])

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(bayes.mu_)):
            fig.add_traces([get_ellipse(bayes.mu_[i], np.diag(bayes.vars_[i])),
                            get_ellipse(lda.mu_[i], lda.cov_)],
                           rows=[1, 1], cols=[1, 2])

        fig.update_layout(title_text=f"<b>Comparing Gaussian Classifiers over the {f.split('.')[0]} dataset</b>",
                          showlegend=False, title_font_size=30, margin=dict(t=75))
        fig.update_annotations(font_size=25)
        fig.update_xaxes(title_text='x1', title_font_size=15)
        fig.update_yaxes(title_text='x2', title_font_size=15)

        save_figure(fig, 'compare_gaussian_classifiers_{}.html'.format(f.split('.')[0]), FIGS_DIR)


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
