import calplot  # TODO: this needs to be installed as a dependency
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from village.classes.plot import SubjectPlot


class Subject_Plot(SubjectPlot):
    def __init__(self) -> None:
        super().__init__()

    def create_plot(self, df: pd.DataFrame) -> Figure:
        """
        Overrides the default method to add a calendar
        """
        dates_df = df.date.value_counts(sort=False)
        dates_df.index = pd.to_datetime(dates_df.index)

        # make the calendar plot and convert it to an image
        cpfig,_ = calplot.calplot(data=dates_df)
        canvas = FigureCanvasAgg(cpfig)
        canvas.draw()
        width, height = cpfig.get_size_inches() * cpfig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        plt.close(cpfig)

        # create the main figure
        fig, axs = plt.subplots(2,1,figsize=(10, 6))
        axs[0].imshow(image)
        axs[0].axis('off')
        dates_df.plot(kind="bar", ax=axs[1])
        axs[1].set_ylabel("Number of trials")

        return fig
        