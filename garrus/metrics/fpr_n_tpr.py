import numpy as np
from sklearn import metrics

from garrus.core import BaseMetric


class FPRnTPR(BaseMetric):
    """False positive rate at n% true positive rate."""

    def _compute(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs: float) -> float:
        """
        Keyword Args:
            n (float [0, 1]): true positive rate %.
        """
        fpr, tpr, thresholds = metrics.roc_curve(accuracies, confidences)
        idx_tpr = np.argmin(np.abs(tpr - kwargs["n"]))  # noqa
        fpr_n_tpr = fpr[idx_tpr]

        return float(fpr_n_tpr)
