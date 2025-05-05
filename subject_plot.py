import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from village.classes.plot import SubjectPlotFigureManager

class SubjectPlot(SubjectPlotFigureManager):
    def __init__(self) -> None:
        super().__init__()
        
    def create_plot(
        self,
        df: pd.DataFrame,
        width: float = 10,
        height: float = 8,
    ) -> Figure:
        """
        Number of trials and calendar.
        """
        # Get counts of trials per date
        dates_df = df.date.value_counts(sort=False)
        dates_df.index = pd.to_datetime(dates_df.index)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height))
        
        # In the first subplot, create a heatmap-style calendar view
        # Group by year and month to create a calendar-like view
        dates_df_monthly = dates_df.groupby([dates_df.index.year, dates_df.index.month]).sum()
        
        # Create a matrix for the heatmap
        years = dates_df.index.year.unique()
        months = range(1, 13)
        
        heat_data = np.zeros((len(years), 12))
        for i, year in enumerate(years):
            for j, month in enumerate(months):
                if (year, month) in dates_df_monthly.index:
                    heat_data[i, j] = dates_df_monthly.loc[(year, month)]
        
        # Plot heatmap
        im = ax1.imshow(heat_data, cmap='viridis')
        ax1.set_title('Trial Activity Calendar')
        
        # Add year labels on y-axis
        ax1.set_yticks(range(len(years)))
        ax1.set_yticklabels(years)
        
        # Add month labels on x-axis
        ax1.set_xticks(range(12))
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        # Add a colorbar
        cbar = fig.colorbar(im, ax=ax1)
        cbar.set_label('Number of trials')
        
        # Create bar chart on second subplot
        dates_df.plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Number of trials")
        ax2.set_title("Trials per day")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        return fig