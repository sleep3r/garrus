from typing import Union
from abc import ABC, abstractmethod

import numpy as np


class BaseVisualization(ABC):
    def __init__(self, n_bins: int, plot_h: int = 650, plot_w: int = 800):
        self.n_bins = n_bins

        self.plot_h = plot_h
        self.plot_w = plot_w

    @abstractmethod
    def _plot(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs: Union[int, float]) -> None:
        """
        Private visualization plotting method.

        Args:
             confidences (np.ndarray): array of confidence levels for each sample;
             accuracies (np.ndarray): array of 0/1 labels of correctness for each sample;
             kwargs (Any): methods parameters.
        """

    def plot(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs: Union[int, float]) -> None:
        """
        Main visualization plotting method.

        Args:
             confidences (np.ndarray): array of confidence levels for each sample;
             accuracies (np.ndarray): array of 0/1 labels of correctness for each sample;
             kwargs (Any): methods parameters.
        """
        assert confidences.shape[0] == accuracies.shape[0], "Number of conf and acc samples is not equal."
        assert accuracies.size > 0, "No samples provided."

        self._plot(confidences, accuracies, **kwargs)
