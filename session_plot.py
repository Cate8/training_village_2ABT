import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from village.classes.plot import SessionPlot


class Session_Plot(SessionPlot):
    def __init__(self) -> None:
        super().__init__()

    def create_plot(self, df: pd.DataFrame, df_raw: pd.DataFrame) -> Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(kind="bar", x=df.columns[0], y=df.columns[1], ax=ax)
        ax.set_title("Session Plot")
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
        return fig
