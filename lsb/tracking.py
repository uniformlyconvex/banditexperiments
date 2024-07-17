import math
import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go
import typing as t


class _Scalar:
    """This is an absolutely disgusting solution, revisit this"""
    def __init__(
        self,
        cumulative: bool=False,
        average: bool=False,
        max_runs: int | None = None,
        max_time_steps: int | None = None
    ) -> None:
        self.cumulative = cumulative
        self.average = average
        self.max_runs = max_runs
        self.max_time_steps = max_time_steps

        self._raw_values: np.ndarray | None = None

        # I hate this I hate this I hate this
        # This holds a dict of the last index of each run we wrote to,
        # so we can append to the correct run
        self._last_index: dict[int, int] = {}

    def _know_size(self) -> bool:
        return self.max_runs is not None and self.max_time_steps is not None
    
    @property
    def raw_values(self) -> np.ndarray:
        return self._raw_values

    def append_value(self, value: float, run: int) -> None:
        if self._raw_values is None:
            if self._know_size():
                self._raw_values = np.full((self.max_runs, self.max_time_steps), np.nan)
                self._raw_values[run, 0] = value
            else:
                self._raw_values = np.array([[value]])
            self._last_index[run] = 0
            return
        
        if run not in self._last_index and run != 0 and not self._know_size():
            # We're starting a new run, so we need to pad the array
            # with NaNs in the new row
            self._raw_values = np.pad(
                self._raw_values,
                ((0, 1), (0,0)),
                mode='constant',
                constant_values=np.nan
            )
        next_index = self._last_index.get(run, -1) + 1
        if not self._know_size() and (next_index > self._raw_values.shape[1] - 1):  # Minus one because of counting from 0
            self._raw_values = np.pad(
                self._raw_values,
                ((0, 0), (0, 1)),
                mode='constant',
                constant_values=np.nan
            )
        self._raw_values[run, next_index] = value
        self._last_index[run] = next_index

    def for_plotting(self) -> np.ndarray:
        values = self._raw_values
        if self.cumulative:
            values = np.cumsum(values, axis=1)
        if self.average:
            values = np.mean(values, axis=0)
        return values


class Tracker:
    """Keeps track of some values over time"""
    def __init__(self, max_runs: int | None = None, max_time_steps: int | None = None) -> None:
        self.max_runs = max_runs
        self.max_time_steps = max_time_steps
        self._scalars: dict[str, _Scalar] = {}
        self._reset_runs()

    def new_run(self) -> None:
        if not self._ready_for_new_run:
            raise ValueError(
                "Nothing has been tracked since the last run;"
                "you called new_run multiple times in a row"
            )
        self._run_number += 1
        self._ready_for_new_run = False

    def finished_runs(self) -> None:
        self._reset_runs()

    def _reset_runs(self) -> None:
        self._run_number = -1
        self._ready_for_new_run = True

    def track_scalars(
        self,
        values: dict[str, float | int],
        cumulative: bool=False,
        average_over_runs: bool=False
    ) -> None:
        for name, value in values.items():
            if name not in self._scalars:
                self._scalars[name] = _Scalar(
                    cumulative=cumulative,
                    average=average_over_runs,
                    max_runs=self.max_runs,
                    max_time_steps=self.max_time_steps
                )
            self._scalars[name].append_value(value, self._run_number)
            self._scalars[name].cumulative = cumulative
            self._scalars[name].average = average_over_runs

        # We've done something now, so we're ready for a new run
        self._ready_for_new_run = True

    def plot_all(self, n_rows: int | None = None, n_cols: int | None = None) -> None:
        if n_rows is None and n_cols is None:
            n_cols = n_rows = math.ceil(math.sqrt(len(self._scalars)))

        if n_rows * n_cols < len(self._scalars):
            raise ValueError("Not enough subplots for all scalars")

        figure = sp.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(self.scalars.keys()))

        for i, (name, scalar) in enumerate(self._scalars.items()):
            values = scalar.for_plotting()
            trace = go.Scatter(
                y=values,
                name=name,
                mode='lines'
            )
            row_no = i // n_cols + 1
            col_no = i % n_cols + 1

            figure.update_xaxes(title_text='Time', row=row_no, col=col_no)
            figure.update_yaxes(title_text='Value', row=row_no, col=col_no)
            figure.add_trace(trace, row_no, col_no)
        figure.show()

    def plot_together(
        self,
        groups: t.Iterable[t.Iterable[str]],
    ) -> None:
        n_cols = n_rows = math.ceil(math.sqrt(len(groups)))

        figure = sp.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[', '.join(group) for group in groups])

        for i, group in enumerate(groups):
            for name in group:
                history = self._scalars[name].for_plotting()
                trace = go.Scatter(y=history, name=name, mode='lines')
                row_no = i // n_cols + 1
                col_no = i % n_cols + 1

                figure.update_xaxes(title_text='Time', row=row_no, col=col_no)
                figure.update_yaxes(title_text='Value', row=row_no, col=col_no)
                figure.add_trace(trace, row_no, col_no)
        figure.show()