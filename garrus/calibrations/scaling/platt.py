from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from torch.nn import Softmax

from garrus.const import EPS
from garrus.core import BaseCalibration
from garrus.metrics import NLL


class Platt(BaseCalibration):
    def __init__(self):
        super().__init__()

    def _fit(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs) -> Platt:
        X = self._scipy_transform_data(confidences)

        weights_num = 2
        x0 = np.ones(weights_num)
        x0[0] = EPS

        result = minimize(fun=self._loss_func, x0=x0, args=(X, accuracies))

        self._weights = result.x
        return self

    def _transform(self, confidences: np.ndarray, **kwargs) -> np.ndarray:
        X = self._scipy_transform_data(confidences)
        logits = self._update_weights(X, self._weights)
        calibrated = Softmax(dim=1)(logits).numpy()
        return calibrated

    def _loss_func(self, weights: np.ndarray, X: np.ndarray, y: np.ndarray):
        logits = self._update_weights(X, weights)
        return NLL().compute(logits, y)
