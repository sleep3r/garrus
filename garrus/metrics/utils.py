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