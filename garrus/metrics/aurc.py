import numpy as np

from garrus.core import BaseMetric


def coverage_risk(confidences, accuracies):
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidences)):
        coverage = (i + 1) / len(confidences)
        coverage_list.append(coverage)

        if accuracies[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))
    return risk_list, coverage_list


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
        return float(aurc)
