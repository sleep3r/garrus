import numpy as np

from garrus.core import BaseMetric


class Brier(BaseMetric):
    """
    Brier score.

    $$ Brier = -\frac{1}{m} \sum_{j=1}^{m} (y_{j}-b_{j})^{2}) $$
    """

    def _compute(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs) -> float:
        brier_score = np.mean(np.sum((confidences - accuracies) ** 2, axis=1))  # noqa
        return float(brier_score)
