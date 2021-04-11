import numpy as np

from garrus.core import BaseMetric
from garrus.metrics.aurc import coverage_risk


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
        return float(eaurc)
