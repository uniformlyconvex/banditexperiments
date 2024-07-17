import math
import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go
import typing as t


class Tracker:
    """Keeps track of some values over time"""
    def __init__(self) -> None:
        self._scalars: dict[str, np.ndarray] = {}

    @property
    def scalars(self) -> dict[str, np.ndarray]:
        return self._scalars

    def track_scalars(self, values: dict[str, float | int], cumulative: bool=False) -> None:
        for name, value in values.items():
            if name not in self._scalars:
                self._scalars[name] = np.array([value])
            else:
                if cumulative:
                    value += self._scalars[name][-1]
                self._scalars[name] = np.append(self._scalars[name], value)


    def plot_all(self, n_rows: int | None = None, n_cols: int | None = None) -> None:
        if n_rows is None and n_cols is None:
            n_cols = n_rows = math.ceil(math.sqrt(len(self._scalars)))

        if n_rows * n_cols < len(self._scalars):
            raise ValueError("Not enough subplots for all scalars")

        figure = sp.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(self.scalars.keys()))

        for i, (name, history) in enumerate(self._scalars.items()):
            trace = go.Scatter(y=history, name=name, mode='lines')
            row_no = i // n_cols + 1
            col_no = i % n_cols + 1

            figure.update_xaxes(title_text='Time', row=row_no, col=col_no)
            figure.update_yaxes(title_text='Value', row=row_no, col=col_no)
            figure.add_trace(trace, row_no, col_no)
        figure.show()

    def plot_together(self, groups: t.Iterable[t.Iterable[str]]) -> None:
        n_cols = n_rows = math.ceil(math.sqrt(len(groups)))

        figure = sp.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[', '.join(group) for group in groups])

        for i, group in enumerate(groups):
            for name in group:
                history = self._scalars[name]
                trace = go.Scatter(y=history, name=name, mode='lines')
                row_no = i // n_cols + 1
                col_no = i % n_cols + 1

                figure.update_xaxes(title_text='Time', row=row_no, col=col_no)
                figure.update_yaxes(title_text='Value', row=row_no, col=col_no)
                figure.add_trace(trace, row_no, col_no)
        figure.show()