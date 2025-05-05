import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from village.classes.plot import SessionPlotFigureManager

class SessionPlot(SessionPlotFigureManager):
    def __init__(self) -> None:
        super().__init__()

    def create_plot(
        self,
        df: pd.DataFrame,
        width: float = 10,
        height: float = 8,
    ) -> Figure:
        """
        Cumulative count of trials by trial start.
        """
        task = df.task.iloc[0]

        if task == "Habituation":
            return self.plot_Habituation(df, width, height)
        elif task == "S1":
            return self.plot_S1(df, width, height)
        elif task == "S2":
            return self.plot_S2(df, width, height)
        elif task == "S3":
            return self.plot_S3(df, width, height)
        elif task == "S4":
            return self.plot_S4(df, width, height)
        

    def plot_Habituation(self, df: pd.DataFrame, width: float = 10, height: float = 8) -> Figure:
        fig, ax = plt.subplots(figsize=(width, height))
        df.plot(kind="line", x="TRIAL_START", y="trial", ax=ax)
        ax.scatter(df["TRIAL_START"], df["trial"], color="blue")
        return fig
    
    def plot_S1(self, df: pd.DataFrame, width: float = 10, height: float = 8) -> Figure:
        fig, ax = plt.subplots(figsize=(width, height))
        df.plot(kind="line", x="TRIAL_START", y="trial", ax=ax)
        ax.scatter(df["TRIAL_START"], df["trial"], color="green")
        return fig
    
    def plot_S2(self, df: pd.DataFrame, width: float = 10, height: float = 8) -> Figure:
        fig, ax = plt.subplots(figsize=(width, height))
        df.plot(kind="line", x="TRIAL_START", y="trial", ax=ax)
        ax.scatter(df["TRIAL_START"], df["trial"], color="orange")
        return fig
    
    def plot_S3(self, df: pd.DataFrame, width: float = 10, height: float = 8) -> Figure:
        fig, ax = plt.subplots(figsize=(width, height))
        df.plot(kind="line", x="TRIAL_START", y="trial", ax=ax)
        ax.scatter(df["TRIAL_START"], df["trial"], color="black")
        return fig
    
    def plot_S4(self, df: pd.DataFrame, width: float = 10, height: float = 8) -> Figure:
        fig, ax = plt.subplots(figsize=(width, height))
        df.plot(kind="line", x="TRIAL_START", y="trial", ax=ax)
        ax.scatter(df["TRIAL_START"], df["trial"], color="purple")
        return fig
