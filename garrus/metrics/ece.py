import numpy as np

from garrus.core import BaseMetric


class ECE(BaseMetric):
    """
    Expected Calibration Error.

    $$ ECE=\frac{1}{|\{B\}| m} \sum_{B \in\{B\}} \mid \sum_{x \in B} b_{k}(x)-\sum_{x \in B} I[y(x)=k] $$,
    for fixed class k.
    """

    def _compute(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs) -> float:
        bins = np.linspace(0, 1, self.n_bins + 1)  # noqa
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]

        ece = np.zeros(1)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower.item()) * (confidences <= bin_upper.item())
            prop_in_bin = in_bin.astype(float).mean()

            if prop_in_bin.item() > 0.0:
                accuracy_in_bin = accuracies[in_bin].astype(float).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece.item())
