import numpy as np
from scipy.special import expit

from garrus.const import EPS
from garrus.core import BaseMetric


class NLL(BaseMetric):
    """
    Negative Log-Likelihood.

    $$ NLL = -\frac{1}{m} \sum_{j=1}^{m} y_{j} \log b_{j} $$
    """

    def _compute(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs) -> float:
        if confidences.ndim == 2:
            confidences = confidences[:, -1]

        confidences = expit(confidences)
        confidences = np.clip(confidences, EPS, 1. - EPS)  # noqa

        if confidences.ndim == 2:
            confidences = np.array(confidences[:, -1])

        cross_entropy = np.multiply(accuracies, np.log(confidences)) + \
                        np.multiply(1. - accuracies, np.log(1 - confidences))

        log_likelihood = np.sum(cross_entropy)  # noqa
        nll = -log_likelihood / accuracies.shape[0]
        return nll
