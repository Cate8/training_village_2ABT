import pandas as pd
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from village.classes.plot import SubjectPlotFigureManager
from plotting_functions import *
from session_parsing_functions import *
from subject_plotting_function import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D
import ast

class SubjectPlot(SubjectPlotFigureManager):
    def __init__(self) -> None:
        super().__init__()

    def create_plot(self, df: pd.DataFrame, width: float = 10, height: float = 8) -> Figure:
        plt.rcParams.update({'font.size': 6, 'font.family': 'monospace'})
        fig = plt.figure(figsize=(width, height))
        df = assign_ports(df)
  
        gs = gridspec.GridSpec(
            3, 3,
            figure=fig,
            height_ratios=[1, 1, 1],
            width_ratios=[1, 1, 1]
        )
        print(df.columns)

        # --- Plot 1: Activity timeline 
        ax0 = fig.add_subplot(gs[0, 0])
        plot_number_of_sessions(df, ax0)

        # --- Plot 2: Water amount
        ax1 = fig.add_subplot(gs[0, 1])
        plot_trials_and_water(df, ax1)  

        # --- Plot 3: Daily Weight 
        ax2 = fig.add_subplot(gs[0, 2])
        
        # --- Plot 4: Cumulative trial progression 
        ax3 = fig.add_subplot(gs[1, 0])
        plot_cumulative_trial_rate(df, ax3)

        # --- Plot 5: Session count per task
        ax4 = fig.add_subplot(gs[1, 1])
        plot_session_count_by_task(ax4, df)

        # --- Plot 6: Task histogram 
        ax5 = fig.add_subplot(gs[1, 2])
        plot_task_histogram(ax5, df)

        # --- Plot 7: Calendar 
        ax6 = fig.add_subplot(gs[2, :])
        plot_calendar(ax6, df)

        plt.subplots_adjust(hspace=0.6, wspace=0.2, top=0.95, bottom=0.09, left= 0.05)
        return fig


    