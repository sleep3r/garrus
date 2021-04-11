from typing import Union, Optional
from abc import ABC, abstractmethod

import numpy as np


class BaseMetric(ABC):
    def __init__(self, n_bins: Optional[int] = None):
        self.n_bins = n_bins

    @abstractmethod
    def _compute(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs) -> float:
        """
        Private metric computation method.

        Args:
             confidences (np.ndarray): array of confidence levels for each sample;
             accuracies (np.ndarray): array of 0/1 labels of correctness for each sample;
             kwargs (Any): methods parameters.
        Returns:
            float: calculated metric.
        """

    def compute(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs: Union[int, float]) -> float:
        """
        Main metric computation method.

        Args:
             confidences (np.ndarray): array of confidence levels for each sample;
             accuracies (np.ndarray): array of 0/1 labels of correctness for each sample;
             kwargs (Any): methods parameters.
        Returns:
            float: calculated metric.
        """
        assert confidences.shape[0] == accuracies.shape[0], "Number of conf and acc samples is not equal."
        assert accuracies.size > 0, "No samples provided."

        metric = self._compute(confidences, accuracies, **kwargs)
        return metric
