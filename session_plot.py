import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker 
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ast
from village.classes.plot import SessionPlotFigureManager
from plotting_functions import *


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

        if task == "S0":
            return self.plot_S0(df, width, height)
        elif task == "S1":
            return self.plot_S1(df, width, height)
        elif task == "S2":
            return self.plot_S2(df, width, height)
        elif task == "S3":
            return self.plot_S3(df, width, height)
        elif task == "S4_0":
            return self.plot_S4(df, width, height)
        

    def plot_S0(self, df: pd.DataFrame, width: float = 8, height: float = 6) -> Figure:
        fig, ax = plt.subplots(figsize=(width, height))
        df.plot(kind="line", x="TRIAL_START", y="trial", ax=ax)
        ax.scatter(df["TRIAL_START"], df["trial"], color="blue")
        return fig
    
    def plot_S1(self, df: pd.DataFrame, width: float = 2, height: float = 2) -> Figure:
        df = assign_ports(df)
        df = parse_data(df)
        
        # --- prepare the grid ---
        fig = plt.figure(figsize=(12, 10), dpi=70)
        gs = gridspec.GridSpec(
            3, 3, 
            figure=fig,
            height_ratios=[0.1, 1, 1],
            width_ratios=[1, 1, 0.3]  
        )

        # === TEXT SUMMARY ===
        ax0 = fig.add_subplot(gs[0, :])  # Summary 
        ax1 = fig.add_subplot(gs[1, 0])  # First poke (side)
        ax2 = fig.add_subplot(gs[1, 1])  # Accuracy
        ax3 = fig.add_subplot(gs[2, 0])  # Trial progression
        ax4 = fig.add_subplot(gs[2, 1])  # Reaction time
        ax5 = fig.add_subplot(gs[1:, 2])  # Lick raster
        n_trials = len(df)
        n_correct = df['correct_outcome_int'].sum()
        pct_correct = round(n_correct / n_trials * 100, 2)
        n_left = (df['first_trial_response'] == 'left').sum()
        n_right = (df['first_trial_response'] == 'right').sum()
        n_omit = (df['first_trial_response'] == 'no_response').sum()
        rt_median = round(df['reaction_time'].median(), 2)
        session_duration_min = round(df['session_duration'].iloc[0], 1)
        summary_text = (
            f"Total trials: {n_trials} | Session: {session_duration_min} min | "
            f"Correct: {n_correct} ({pct_correct}%) | Left: {n_left} | Right: {n_right} | "
            f"Omissions: {n_omit} | Median RT: {rt_median} s"
        )
        ax0.axis("off")
        ax0.text(0, 0.5, summary_text, fontsize=11, va='center', ha='left', family='monospace')
        
        # --- PLOT 1: First poke side by outcome ---
        plot_first_poke_side(ax1, df)

        # --- PLOT 2: Trial progression over time ---
        plot_trial_progression(ax3, df)

       # --- PLOT 3: Accuracy on trial ---
        plot_rolling_accuracy(ax2, df)

        # --- PLOT 4: Reaction time  ---
        plot_reaction_time(ax4, df)

        #--- PLOT 5: Lick Raster  ---
        plot_lick_raster_with_states(ax5, df, fig)

        handles = [
            mpatches.Patch(color='orange', alpha=0.3, label='LED ON'),
            mpatches.Patch(color='gray', alpha=0.3, label='Drink delay'),
            mpatches.Patch(color='lightblue', alpha=0.3, label='Water delivery'),
            mpatches.Patch(color='green', label='Left lick'),
            mpatches.Patch(color='purple', label='Right lick'),
        ]
        #fig.legend( handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1),fontsize=9,frameon=False)
        # --- Adjust layout ---
        plt.subplots_adjust(hspace=0.6, wspace=0.4)
        return fig

    #============================================   S2   =================================================
    def plot_S2(self, df: pd.DataFrame, width: float = 10, height: float = 8) -> Figure:
        df = assign_ports(df)
        df = parse_data(df)
        # --- prepare the grid ---
        fig = plt.figure(figsize=(12, 10), dpi=70)
        gs = gridspec.GridSpec(
            3, 3, 
            figure=fig,
            height_ratios=[0.1, 1, 1],
            width_ratios=[1, 1, 0.3]  
        )
        # === TEXT SUMMARY ===
        ax0 = fig.add_subplot(gs[0, :])  # Summary 
        ax1 = fig.add_subplot(gs[1, 0])  # First poke (side)
        ax2 = fig.add_subplot(gs[1, 1])  # Accuracy
        ax3 = fig.add_subplot(gs[2, 0])  # Trial progression
        ax4 = fig.add_subplot(gs[2, 1])  # Reaction time
        ax5 = fig.add_subplot(gs[1:, 2])  # Lick raster
        n_trials = len(df)
        n_correct = df['correct_outcome_int'].sum()
        pct_correct = round(n_correct / n_trials * 100, 2)
        n_left = (df['first_trial_response'] == 'left').sum()
        n_right = (df['first_trial_response'] == 'right').sum()
        n_omit = (df['first_trial_response'] == 'no_response').sum()
        rt_median = round(df['reaction_time'].median(), 2)
        session_duration_min = round(df['session_duration'].iloc[0], 1)
        summary_text = (
            f"Total trials: {n_trials} | Session: {session_duration_min} min | "
            f"Correct: {n_correct} ({pct_correct}%) | Left: {n_left} | Right: {n_right} | "
            f"Omissions: {n_omit} | Median RT: {rt_median} s"
        )
        ax0.axis("off")
        ax0.text(0, 0.5, summary_text, fontsize=11, va='center', ha='left', family='monospace')
        
        # --- PLOT 1: First poke side by outcome ---
        plot_first_poke_side(ax1, df)

        # --- PLOT 2: Trial progression over time ---
        plot_trial_progression(ax3, df)

       # --- PLOT 3: Accuracy on trial ---
        plot_rolling_accuracy(ax2, df)

        # --- PLOT 4: Reaction time  ---
        plot_reaction_time(ax4, df)

        # --- PLOT 4: Reaction time  ---
        ax = ax4
        ax.plot(df['trial'], df['reaction_time'], color='dodgerblue', linewidth=2, markersize=8)
        # Set Y axis to log scale and format ticks
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        # Title and labels
        ax.set_title('Reaction time (RT)')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Latency to poke (s)')
        # Custom horizontal gridlines at specific y-values
        custom_yticks = [1, 10, 100]
        for y in custom_yticks:
            ax.axhline(y=y, linestyle='--', color='lightgray', alpha=0.7)

        plot_lick_raster_with_states(ax5, df, fig)

        handles = [
            mpatches.Patch(color='orange', alpha=0.3, label='LED ON'),
            mpatches.Patch(color='gray', alpha=0.3, label='Drink delay'),
            mpatches.Patch(color='lightblue', alpha=0.3, label='Water delivery'),
            mpatches.Patch(color='green', label='Left lick'),
            mpatches.Patch(color='purple', label='Right lick'),
        ]
        #fig.legend( handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1),fontsize=9,frameon=False)
        # --- Adjust layout ---
        plt.subplots_adjust(hspace=0.6, wspace=0.4)
        return fig
    
    #============================================   S3   =================================================
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
