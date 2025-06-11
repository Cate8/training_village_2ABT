import pandas as pd
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from village.classes.plot import SubjectPlotFigureManager
from plotting_functions import *
from session_parsing_functions import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates

class SubjectPlot(SubjectPlotFigureManager):
    def __init__(self) -> None:
        super().__init__()

    def create_plot(self, df: pd.DataFrame, width: float = 10, height: float = 8) -> Figure:
        plt.rcParams.update({'font.size': 6, 'font.family': 'monospace'})
        fig = plt.figure(figsize=(width, height))
        
        gs = gridspec.GridSpec(
            3, 3,
            figure=fig,
            height_ratios=[0.5, 1, 1],
            width_ratios=[1, 1, 0.3]
        )

        # Summary
        ax0 = fig.add_subplot(gs[0, :])  # Summary
        self.plot_activity_timeline(ax0, df)

         # Calendar plot
        ax1 = fig.add_subplot(gs[1, 1])
        self.plot_calendar(ax1, df)

        # First poke (side)
        ax2 = fig.add_subplot(gs[1, 0])
        self.plot_session_count_by_task(ax2, df)

        # Trial progression
        ax3 = fig.add_subplot(gs[2, 0])
        self.plot_task_histogram(ax3, df)

        # # Reaction time
        # ax4 = fig.add_subplot(gs[2, 1])
        # ax4.set_title("Reaction time")

        # # Lick raster
        # ax5 = fig.add_subplot(gs[1:, 2])
        # ax5.set_title("Lick raster")

        plt.subplots_adjust(hspace=0.6, wspace=0.2, top=0.95, bottom=0.09, left= 0.05)
        return fig
        
    def plot_calendar(self, ax, df):
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.date

        grouped = df.groupby(['date', 'task']).size().reset_index(name='count')

        base_colors = {
            'S0':'#cccccc',
            'S1': '#2ecc71',
            'S2': '#3498db',
            'S3': '#9b59b6',
            'S4_0': '#f1c40f',
            'S4_1': '#e67e22',
            'S4_2': '#e74c3c',
            'S4_3': '#c0392b',
        }

        def color_with_alpha(task, count):
            base = mcolors.to_rgba(base_colors.get(task, 'gray'))
            alpha = min(1.0, 0.2 + 0.16 * count)
            return (*base[:3], alpha)

        min_date = grouped['date'].min()
        max_date = grouped['date'].max()
        start = min_date - timedelta(days=min_date.weekday())
        end = max_date + timedelta(days=(6 - max_date.weekday()))
        all_days = pd.date_range(start, end, freq='D')

        calendar_data = {}
        for _, row in grouped.iterrows():
            calendar_data[row['date']] = {
                'color': color_with_alpha(row['task'], row['count']),
                'task': row['task'],
                'count': row['count'],
            }

        ax.set_xlim(0, len(all_days) // 7 + 1)
        ax.set_ylim(-0.5, 6.5)
        ax.set_yticks(range(7))
        ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

        for i, single_date in enumerate(all_days):
            week = i // 7
            weekday = single_date.weekday()
            color = calendar_data.get(single_date.date(), {}).get('color', (0.95, 0.95, 0.95, 1))
            ax.add_patch(plt.Rectangle((week, weekday), 1, 1, color=color))

        ax.set_aspect('equal')
        ax.invert_yaxis()
        # Set X ticks to months
        n_weeks = len(all_days) // 7 + 1
        month_positions = {}
        for i, date in enumerate(all_days):
            week = i // 7
            month = date.strftime("%b")
            if week not in month_positions:
                month_positions[week] = month

        ax.set_xticks(list(month_positions.keys()))
        ax.set_xticklabels(list(month_positions.values()))
        ax.set_title("Animal training calendar")

        # handles = [mpatches.Patch(color=base_colors[k], label=k) for k in base_colors]
        # ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=6, title='Task')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def plot_task_histogram(self, ax, df):
        """
        Plot stacked bar histogram of trials per day, grouped by task,
        using consistent task colors.
        """
        import matplotlib.colors as mcolors

        # Ensure date is datetime.date (not datetime64)
        df['date'] = pd.to_datetime(df['date']).dt.date

        # Count trials per (date, task)
        trial_counts = df.groupby(['date', 'task']).size().unstack(fill_value=0)

        # Define consistent task colors
        task_colors = {
            'S0': '#cccccc',
            'S1': '#2ca02c',       # green
            'S2': '#1f77b4',       # blue
            'S3': '#9467bd',       # violet
            'S4_0': '#ffcc66',     # light orange
            'S4_1': '#ff9933',     # medium orange
            'S4_2': '#ff3300',     # red
            'S_3':  '#990000',     # dark red
        }

        # Ensure all tasks are present
        for col in trial_counts.columns:
            if col not in task_colors:
                task_colors[col] = 'gray'

        # Plot
        trial_counts.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            color=[task_colors[task] for task in trial_counts.columns],
            width=0.9,
            edgecolor='white'
        )

        # Formatting
        ax.set_title("Trials per day per task")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of trials")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # # Legend
        # handles = [
        #     mpatches.Patch(color=task_colors[task], label=task)
        #     for task in trial_counts.columns
        # ]
        #ax.legend(handles=handles, title="Task", fontsize=6, loc='upper left', bbox_to_anchor=(1.01, 1.0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def plot_session_count_by_task(self, ax, df):
        """
        Plot histogram showing the number of unique sessions per task type.
        Uses consistent colors with the calendar plot.
        """
        import matplotlib.patches as mpatches

        # Define consistent task colors
        task_colors = {
            'S0': '#cccccc',
            'S1': '#2ca02c',       # green
            'S2': '#1f77b4',       # blue
            'S3': '#9467bd',       # violet
            'S4_0': '#ffcc66',     # light orange
            'S4_1': '#ff9933',     # medium orange
            'S4_2': '#ff3300',     # red
            'S_3':  '#990000',     # dark red
        }

        # Count unique session dates per task
        df['date'] = pd.to_datetime(df['date']).dt.date
        session_counts = df.groupby('task')['date'].nunique().sort_values(ascending=False)

        # Plot bar chart
        session_counts.plot(
            kind='bar',
            ax=ax,
            color=[task_colors.get(task, 'gray') for task in session_counts.index],
            edgecolor='white'
        )

        # Formatting
        ax.set_title("Sessions per task", pad=10)
        ax.set_ylabel("Number of sessions")
        ax.set_xlabel("Task")
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.tick_params(axis='x', rotation=45)

        # Add count labels on top of bars
        for i, val in enumerate(session_counts.values):
            ax.text(i, val + 0.3, str(val), ha='center', va='bottom', fontsize=6)

        # Optional: custom legend (if some tasks are missing)
        handles = [
            mpatches.Patch(color=task_colors[task], label=task)
            for task in session_counts.index
        ]
        ax.legend(handles=handles, title="Task", fontsize=6, loc='upper left', bbox_to_anchor=(1.01, 1.0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


    def plot_activity_timeline(self, ax, df):
        """
        Plot horizontal subject activity (last 7 days).
        Each session is a black horizontal bar.
        Background: white (08–20h), gray (20–08h).
        """
        # Drop NaT or missing times
        df = df.dropna(subset=['TRIAL_START', 'TRIAL_END'])
        df['TRIAL_START'] = pd.to_datetime(df['TRIAL_START'])
        df['TRIAL_END'] = pd.to_datetime(df['TRIAL_END'])
        # Ensure datetime format
        if np.issubdtype(df['TRIAL_START'].dtype, np.number):
            df['TRIAL_START'] = pd.to_datetime(df['TRIAL_START'], unit='s')  # from epoch time
        else:
            df['TRIAL_START'] = pd.to_datetime(df['TRIAL_START'], errors='coerce')

        if np.issubdtype(df['TRIAL_END'].dtype, np.number):
            df['TRIAL_END'] = pd.to_datetime(df['TRIAL_END'], unit='s')
        else:
            df['TRIAL_END'] = pd.to_datetime(df['TRIAL_END'], errors='coerce')
        # Filter to last 7 days from today
        today = pd.Timestamp.now().normalize()
        last_week = today - pd.Timedelta(days=7)
        df = df[df['TRIAL_START'] >= last_week]

        if df.empty:
            ax.text(0.5, 0.5, "No sessions in the last 7 days", ha='center', va='center', fontsize=8)
            ax.axis("off")
            return

        # Sort by start time
        df = df.sort_values(by='TRIAL_START')
        # Background: alternating day (white) and night (gray)
        start_date = df['TRIAL_START'].min().normalize()
        end_date = df['TRIAL_END'].max().normalize() + pd.Timedelta(days=1)
        current = start_date
        while current < end_date:
            day_start = current + pd.Timedelta(hours=8)
            night_start = current + pd.Timedelta(hours=20)
            next_day = current + pd.Timedelta(days=1)

            ax.axvspan(current, day_start, color='gray', alpha=0.2)      # night before 8AM
            ax.axvspan(night_start, next_day, color='gray', alpha=0.2)   # night after 8PM

            current = next_day

        # Plot sessions
        for _, row in df.iterrows():
            ax.plot([row['TRIAL_START'], row['TRIAL_END']], [0.5, 0.5], color='black', linewidth=4)
        
        # Format axes
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlim(start_date, end_date)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.set_title('Subject activity (last 7 days)')
        ax.set_xlabel('Date')
        ax.grid(True, axis='x', linestyle=':', alpha=0.3)


        
