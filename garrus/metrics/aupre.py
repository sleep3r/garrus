import numpy as np
from sklearn import metrics

from garrus.core import BaseMetric


class AUPRE(BaseMetric):
    """Area under the precision-recall curve using errors as the positive class."""

    def _compute(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs) -> float:
        aupr_err = metrics.average_precision_score(-1 * accuracies + 1, -1 * confidences)

        return float(aupr_err)
