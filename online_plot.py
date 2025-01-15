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
        # self.required_columns = ["TRIAL_START", "trial"]
        # self.extra_columns = ["correct"]
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)

    # def create_multiplot(self, df: pd.DataFrame) -> Figure:
    #     try:
    #         self.df = df[self.required_columns].copy()
    #         for extra_column in self.extra_columns:
    #             if extra_column in df.columns:
    #                 self.df[extra_column] = df[extra_column]
    #                 if extra_column == "correct":
    #                     # convert True and False to 1 and 0
    #                     self.df["correct"] = self.df["correct"].apply(
    #                         lambda x: int(ast.literal_eval(x))
    #                     )
    #         self.make_plot()
    #         self.fig.tight_layout()
    #     except Exception:
    #         self.make_error_plot()
    #         print("Could not create plot")

    #     return self.fig

    def update_plot(self, df: pd.DataFrame) -> None:
        try:
            self.make_timing_plot(df, self.ax1)
        except Exception:
            self.make_error_plot(self.ax1)
        try:
            self.make_trial_side_and_correct_plot(df, self.ax2)
        except Exception:
            self.make_error_plot(self.ax2)
        
        self.fig.tight_layout()
        self.fig.canvas.draw()

    def make_timing_plot(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        ax.clear()
        df.plot(kind="scatter", x="TRIAL_START", y="trial", ax=ax)

    def make_trial_side_and_correct_plot(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        _ = plot_side_correct_performance(df, ax)        

    def make_error_plot(self, ax) -> None:
        ax.clear()
        ax.text(
            0.5,
            0.5,
            "Could not create plot",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
