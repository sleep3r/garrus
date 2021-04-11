from typing import Union

import numpy as np
import plotly.graph_objects as go

from garrus.core import BaseVisualization


class ConfidenceHistogram(BaseVisualization):
    def _plot(self, confidences: np.ndarray, accuracies: np.ndarray, **kwargs: Union[int, float]) -> None:
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=confidences,
                xbins=dict(start=0, end=1.0, size=1 / self.n_bins),
                name="Amount of samples"
            )
        )

        fig.add_vline(
            x=confidences.mean(),
            line_width=3, line_dash="dash", line_color="green",
            name="Avg. confidence"
        )
        fig.add_trace(
            go.Scatter(
                x=[confidences.mean()], y=[0], mode="markers", name="Avg. confidence",
                marker={"color": "green", "size": 10}
            )
        )

        fig.add_vline(
            x=accuracies.mean(),
            line_width=3, line_dash="dash", line_color="red",
            name="Avg. accuracy"
        )
        fig.add_trace(
            go.Scatter(
                x=[accuracies.mean()], y=[0], mode="markers", name="Avg. accuracy",
                marker={"color": "red", "size": 10}
            )
        )

        fig.update_yaxes(rangemode="tozero")

        fig.update_layout(
            title="Confidence Histogram",
            xaxis_title="Confidence",
            yaxis_title="Samples",
            showlegend=True,
            height=self.plot_h, width=self.plot_w,
        )

        fig.show()
