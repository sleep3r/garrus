import numpy as np

from garrus.core import BaseMetric


class NLL(BaseMetric):
    """
    Negative Log-Likelihood.

    $$ NLL = -\frac{1}{m} \sum_{j=1}^{m} y_{j} \log b_{j} $$
    """

    def _compute(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs) -> float:
        log_conf = np.log(confidences)
        res = np.zeros_like(accuracies, dtype=float)

        for i in range(accuracies.shape[0]):
            res[i] = log_conf[i][accuracies[i]]

        nll = -res.sum() / res.shape[0]
        print("NLL {0:.3f} ".format(nll.item() * 10))
        return float(nll.item())
