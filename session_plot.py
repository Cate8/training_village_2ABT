import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from village.classes.plot import SessionPlotFigureManager

class SessionPlot(SessionPlotFigureManager):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Session Plot"

    def create_plot(
        self,
        df: pd.DataFrame,
        width: float = 10,
        height: float = 8,
    ) -> Figure:
        """
        Cumulative count of trials by trial start.
        """
        fig, ax = plt.subplots(figsize=(width, height))
        df.plot(kind="line", x="TRIAL_START", y="trial", ax=ax)
        ax.scatter(df["TRIAL_START"], df["trial"], color="red")
        return fig
