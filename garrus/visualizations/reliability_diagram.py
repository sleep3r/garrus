from typing import Union, Tuple, List

import numpy as np
import plotly.graph_objects as go

from garrus.core import BaseVisualization


class ReliabilityDiagram(BaseVisualization):
    def __calc_statistics(
            self, confidences: np.ndarray, accuracies: np.ndarray
    ) -> Tuple[List[float], List[float], List[float]]:
        mean_conf_by_bin: List[float] = []
        mean_acc_by_bin: List[float] = []
        samples_pct: List[float] = []

        bin_idxs = np.digitize(confidences, np.histogram_bin_edges(confidences, bins=self.n_bins))

        for bin_idx in range(1, 11):
            mean_conf_by_bin.append(confidences[bin_idxs == bin_idx].mean() \
                                        if any(confidences[bin_idxs == bin_idx]) else 0)
            mean_acc_by_bin.append(accuracies[bin_idxs == bin_idx].mean() \
                                       if any(accuracies[bin_idxs == bin_idx]) else 0)
            samples_pct.append((bin_idxs == bin_idx).sum() / bin_idxs.shape[0])

        return mean_conf_by_bin, mean_acc_by_bin, samples_pct

    def _plot(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs: Union[int, float]) -> None:
        mean_conf_by_bin, mean_acc_by_bin, samples_pct = self.__calc_statistics(confidences, accuracies)

        bars = [x / self.n_bins for x in range(self.n_bins + 1)]

        fig = go.Figure()

        # confidence --------------
        fig.add_trace(
            go.Bar(
                y=mean_conf_by_bin, x=bars, name="Mean confidence",
                marker={"color": "red", "opacity": 0.6}
            ),
        )

        # accuracy ----------------
        fig.add_trace(
            go.Bar(
                y=mean_acc_by_bin, x=bars, name="Mean accuracy",
                marker={"color": "blue", "opacity": 0.6}
            ),
        )
        fig.add_trace(
            go.Scatter(
                y=mean_acc_by_bin, x=bars, name="Mean confidence dot",
                marker={"color": "blue", "size": 9}, mode="markers+lines", showlegend=False
            ),
        )

        # % of samples ------------
        fig.add_trace(
            go.Bar(
                y=samples_pct, x=bars, name="% of samples", width=0.025,
                marker={"color": "black", "opacity": 1.0}
            ),
        )

        # ideal line --------------
        fig.add_shape(
            type="line",
            x0=-0.05, y0=0, x1=0.95, y1=1,
            line=dict(
                color="black",
                width=2,
                dash="dot",
            )
        )

        fig.update_yaxes(rangemode="tozero")

        fig.update_layout(
            title="Reliability Plot",
            xaxis_title="Confidence",
            yaxis_title="Accuracy",
            showlegend=True,
            barmode="overlay",
            bargap=0,
            height=self.plot_h, width=self.plot_w,
        )

        fig.show()
