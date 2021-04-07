import numpy as np
from sklearn import metrics

from garrus.core import BaseMetric
from garrus.metrics.utils import coverage_risk


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

        print("ECE {0:.3f} ".format(ece.item() * 100))
        return float(ece.item())


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


class Brier(BaseMetric):
    """
    Brier score.

    $$ Brier = -\frac{1}{m} \sum_{j=1}^{m} (y_{j}-b_{j})^{2}) $$
    """

    def _compute(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs) -> float:
        brier_score = np.mean(np.sum((confidences - accuracies) ** 2, axis=1))  # noqa
        print('Brier {0:.2f}'.format(brier_score * 100))
        return float(brier_score)


class AURC(BaseMetric):
    """
    Area Under Risk-Coverage.

    $$ AURC (\kappa, f \mid V_{n}) = \frac{1}{n} \sum_{\theta \in \Theta} \hat{r} (f, g_{\theta} \mid V_{n}) $$
    """

    def _compute(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs) -> float:
        sort_values = sorted(zip(confidences, accuracies), key=lambda x: x[0], reverse=True)
        sort_conf, sort_acc = zip(*sort_values)
        risk_list, coverage_list = coverage_risk(sort_conf, sort_acc)

        risk_coverage_curve_area = 0
        for risk_value in risk_list:
            risk_coverage_curve_area += risk_value * (1 / len(risk_list))

        aurc = risk_coverage_curve_area
        print("AURC {0:.2f}".format(aurc * 1000))
        return float(aurc)


class EAURC(BaseMetric):
    """
    Excess Area Under Risk-Coverage.

    $$ E-AURC = AURC (\kappa, f \mid V_{n}) - AURC (\kappa^{*}, f \mid V_{n}) $$
    """

    def _compute(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs) -> float:
        sort_values = sorted(zip(confidences, accuracies), key=lambda x: x[0], reverse=True)
        sort_conf, sort_acc = zip(*sort_values)
        risk_list, coverage_list = coverage_risk(sort_conf, sort_acc)

        r = risk_list[-1]
        risk_coverage_curve_area = 0
        optimal_risk_area = r + (1 - r) * np.log(1 - r)
        for risk_value in risk_list:
            risk_coverage_curve_area += risk_value * (1 / len(risk_list))

        eaurc = risk_coverage_curve_area - optimal_risk_area
        print("EAURC {0:.2f}".format(eaurc * 1000))
        return float(eaurc)


class AUPRE(BaseMetric):
    """Area under the precision-recall curve using errors as the positive class."""

    def _compute(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs) -> float:
        aupr_err = metrics.average_precision_score(-1 * accuracies + 1, -1 * confidences)

        print("AUPR {0:.2f}".format(aupr_err * 100))
        return float(aupr_err)


class FPR_n_TPR(BaseMetric):
    """False positive rate at n% true positive rate."""

    def _compute(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs: float) -> float:
        """
        Keyword Args:
            n (float [0, 1]): true positive rate %.
        """
        fpr, tpr, thresholds = metrics.roc_curve(accuracies, confidences)
        idx_tpr = np.argmin(np.abs(tpr - kwargs["n"]))  # noqa
        fpr_n_tpr = fpr[idx_tpr]

        print('FPR {0:.2f}'.format(fpr_n_tpr * 100))
        return float(fpr_n_tpr)
