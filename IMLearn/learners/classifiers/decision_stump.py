from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


BINARY_LABELS = [-1, 1]
class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_loss = np.inf
        min_thr = 0
        min_sign = 0
        min_feature = 0

        for i, sign in product(range(X.shape[1]), BINARY_LABELS):
            thr, loss = self._find_threshold(X[:, i], y, sign)
            if loss <= min_loss:
                min_loss = loss
                min_thr = thr
                min_sign = sign
                min_feature = i

        self.threshold_, self.j_, self.sign_ = min_thr, min_feature, min_sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        values = X[:, self.j_]
        return np.where(values >= self.threshold_, self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # Sort the values array and get the indices according to the sorting
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_labels = labels[sorted_indices]
        # first threshold is the first value, so all labels ill be given 'sign'
        first_prediction = np.full(len(values), sign)
        cur_loss = np.sum(np.where(np.sign(sorted_labels) != first_prediction, np.abs(sorted_labels), 0))
        losses = np.zeros(len(values) + 1)
        losses[0] = cur_loss

        # change the threshold to the ith value, and update the current loss accordingly (i.e., check if we now
        # misclassify the previous label)
        for i in range(1, len(values) + 1):
            if np.sign(sorted_labels[i - 1]) != -sign:
                cur_loss += np.abs(sorted_labels[i - 1])
            else:
                cur_loss -= np.abs(sorted_labels[i - 1])
            losses[i] = cur_loss

        best_index = np.argmin(losses)
        # add infinity at the end of the values array, in case the best threshold is above the last (largest) value
        sorted_values = np.concatenate([[-np.inf], sorted_values[1:], [np.inf]])
        # the so called 'best_index' is according to the sorted values
        return sorted_values[best_index], losses[best_index]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
