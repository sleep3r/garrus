from __future__ import annotations
from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.special import logit

from garrus.const import EPS


class BaseCalibration(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

        self._weights = None
        self.__fitted = False
        self.__accuracies = np.array([])

    @abstractmethod
    def _fit(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs) -> BaseCalibration:
        """
        Private calibration fit method.

        Args:
             confidences (np.ndarray): array of confidence levels for each sample;
             accuracies (np.ndarray): array of 0/1 labels of correctness for each sample;
             kwargs (Any): methods parameters.
        """

    def fit(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs) -> BaseCalibration:
        """
        Main calibration fit method.

        Args:
             confidences (np.ndarray): array of confidence levels for each sample;
             accuracies (np.ndarray): array of 0/1 labels of correctness for each sample;
             kwargs (Any): methods parameters.
        """
        assert confidences.shape[0] == accuracies.shape[0], "Number of conf and acc samples is not equal."
        assert accuracies.size > 0, "No samples provided."
        self.__accuracies = accuracies

        return self._fit(confidences, accuracies, **kwargs)

    @abstractmethod
    def _transform(self, confidences: np.ndarray, **kwargs) -> np.ndarray:
        """
        Private calibration transform method.

        Args:
             confidences (np.ndarray): array of confidence levels for each sample;
             kwargs (Any): methods parameters.
        """

    def transform(self, confidences: np.ndarray, **kwargs) -> np.ndarray:
        """
        Main calibration transform method.

        Args:
             confidences (np.ndarray): array of confidence levels for each sample;
             kwargs (Any): methods parameters.
        """
        assert self.__fitted, "Should call fit() before making transform"
        assert confidences.shape[0] == self.__accuracies.shape[0], "Number of conf and acc samples is not equal."

        return self._transform(confidences, **kwargs)

    def _scipy_transform_data(self, confidences: np.ndarray) -> np.ndarray:
        """
        Transforms data input for calibrations methods utilize scipy optimizations

        Args:
             confidences (np.ndarray): array of confidence levels for each sample;
        Returns:
            np.ndarray: transformed data for scipy input.
        """
        if confidences.ndim == 1:
            confidences = confidences.reshape(-1, 1)
        confidences = np.clip(confidences, EPS, 1. - EPS)  # noqa
        X = logit(confidences)
        return X

    def _update_weights(self, X: np.ndarray, weights: np.ndarray):
        """
        Performs weights update by given data input and weights.
        """
        bias = weights[0]
        weights = np.array(weights[1:]).reshape(-1, 1)
        logits = np.matmul(X, weights) + bias
        return logits
