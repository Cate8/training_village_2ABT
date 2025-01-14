import ast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from village.classes.plot import OnlinePlotFigureManager

from plotting_functions import plot_side_correct_performance


class Online_Plot(OnlinePlotFigureManager):
    def __init__(self) -> None:
        super().__init__()
        self.required_columns = ["TRIAL_START", "trial"]
        self.extra_columns = ["correct"]
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)

    def create_multiplot(self, df: pd.DataFrame) -> Figure:
        try:
            self.df = df[self.required_columns].copy()
            for extra_column in self.extra_columns:
                if extra_column in df.columns:
                    self.df[extra_column] = df[extra_column]
                    if extra_column == "correct":
                        # convert True and False to 1 and 0
                        self.df["correct"] = self.df["correct"].apply(
                            lambda x: int(ast.literal_eval(x))
                        )
            self.make_plot()
            self.fig.tight_layout()
        except Exception:
            self.make_error_plot()
            print("Could not create plot")

        return self.fig

    def update_plot(self, trial_data: dict) -> None:
        try:
            self.update_df(trial_data)
            self.make_plot()
        except Exception:
            self.make_error_plot()

    def update_df(self, trial_data: dict) -> None:
        # get the same keys from the dictionary
        parsed_trial_data = {k: v for k, v in trial_data.items() if k in self.required_columns}
        for extra_column in self.extra_columns:
            if extra_column in trial_data:
                parsed_trial_data[extra_column] = trial_data[extra_column]
        new_row = pd.DataFrame(
            data=parsed_trial_data, columns=self.df.columns, index=[0]
        )
        if "correct" in new_row.columns:
            new_row["correct"] = int(new_row["correct"].iloc[0])
        self.df = pd.concat([self.df, new_row], ignore_index=True)

    def make_plot(self) -> None:
        self.ax1.clear()
        self.df.plot(kind="scatter", x="TRIAL_START", y="trial", ax=self.ax1)
        if "correct" in self.df.columns:
            ax2 = plot_side_correct_performance(self.df, self.ax2)

        self.fig.canvas.draw()

    def make_error_plot(self) -> None:
        self.ax1.clear()
        self.ax1.text(
            0.5,
            0.5,
            "Could not create plot",
            horizontalalignment="center",
            verticalalignment="center",
            transform=self.ax1.transAxes,
        )
        self.fig.canvas.draw()
