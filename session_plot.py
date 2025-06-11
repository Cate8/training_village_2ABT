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
from session_parsing_functions import *


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
        elif task == "S4_1":
            return self.plot_S4(df, width, height)
        elif task == "S4_2":
            return self.plot_S4(df, width, height)
        elif task == "S4_3":
            return self.plot_S4(df, width, height)
        
    def plot_S0(self, df: pd.DataFrame, width: float = 10, height: float = 8) -> Figure:
        fig, ax = plt.subplots(figsize=(width, height))
        df.plot(kind="line", x="TRIAL_START", y="trial", ax=ax)
        ax.scatter(df["TRIAL_START"], df["trial"], color="blue")
        return fig
    
    def plot_S1(self, df: pd.DataFrame, width: float = 10, height: float = 8) -> Figure:
        df = assign_ports(df)
        df = parse_data_S1_S2(df)
        
        # --- prepare the grid ---
        fig = plt.figure(figsize=(width, height))
        #fig.patch.set_facecolor("blue")

        # change the font in figure
        plt.rcParams.update({'font.size': 6, 'font.family': 'monospace'})

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
        ax0.text(0, 0.5, summary_text, fontsize=8, va='center', ha='left', family='monospace')
        
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
        plt.subplots_adjust(hspace=0.4, wspace=0.15, top=0.95, bottom=0.09, left= 0.05)
        return fig

    #============================================   S2   =================================================
    def plot_S2(self, df: pd.DataFrame, width: float = 10, height: float = 8) -> Figure:
        df = assign_ports(df)
        df = parse_data_S1_S2(df)
        # --- prepare the grid ---
        fig = plt.figure(figsize=(width, height))
        gs = gridspec.GridSpec(
            3, 3, 
            figure=fig,
            height_ratios=[0.1, 1, 1],
            width_ratios=[1, 1, 0.3]  
        )
        # change the font in figure
        plt.rcParams.update({'font.size': 6, 'font.family': 'monospace'})
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
        ax0.text(0, 0.5, summary_text, fontsize=8, va='center', ha='left', family='monospace')

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

        # --- PLOT 5: lick raster with states ---
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
        plt.subplots_adjust(hspace=0.4, wspace=0.15, top=0.95, bottom=0.09, left= 0.05)
        return fig
    
    #============================================   S3   =================================================
    def plot_S3(self, df: pd.DataFrame, width: float = 10, height: float = 8) -> Figure:
        df = assign_ports(df)
        df = parse_S3_data(df)
        # change the font in figure
        plt.rcParams.update({'font.size': 6, 'font.family': 'monospace'})
        # --- prepare the grid ---
        fig = plt.figure(figsize=(width, height))
        gs = gridspec.GridSpec(
            4, 3, 
            figure=fig,
            height_ratios=[0.1, 2, 1, 2],
            width_ratios=[1, 1, 0.3]  
        )
        # === TEXT SUMMARY ===
        ax0 = fig.add_subplot(gs[0, :])  # Summary 
        ax1 = fig.add_subplot(gs[1, :])  # First poke (side)
        ax3 = fig.add_subplot(gs[2, 0])  # Trial progression
        ax4 = fig.add_subplot(gs[2, 1:3])  # Reaction time
        ax5 = fig.add_subplot(gs[3, :])  # Lick raster
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
        ax0.text(0, 0.5, summary_text, fontsize=8, va='center', ha='left', family='monospace')
        
        # --- PLOT 1: First poke side by outcome ---
        plot_right_reward_probability(df, ax1)

        # --- PLOT 2: Trial progression over time ---
        plot_trial_progression(ax3, df)
        
        # --- PLOT 3: Motor and Reaction time ---
        plot_latency_to_first_poke(df, ax4)

        # --- PLOT 5: lick raster and state machine ---
        plot_lick_raster_with_states_S3_S4(ax5, df, fig)
        
        plt.subplots_adjust(hspace=0.4, wspace=0.15, top=0.95, bottom=0.09, left= 0.05)
        return fig

    def plot_S4(self, df: pd.DataFrame, width: float = 10, height: float = 8) -> Figure:
        df = assign_ports(df)
        df = parse_S4_data(df)
        # change the font in figure
        plt.rcParams.update({'font.size': 6, 'font.family': 'monospace'})
        # --- prepare the grid ---
        fig = plt.figure(figsize=(width, height))
        gs = gridspec.GridSpec(
            4, 3,  
            figure=fig,
            height_ratios=[0.1, 2, 1, 2], 
            width_ratios=[1, 1, 1]        
)
        # === TEXT SUMMARY ===
        ax0 = fig.add_subplot(gs[0, :])       # Summary
        ax1 = fig.add_subplot(gs[1, 0:2])     # First poke (side)
        ax6 = fig.add_subplot(gs[1, 2])       # Psychometric
        ax3 = fig.add_subplot(gs[2, 0])       # Trial progression
        ax4 = fig.add_subplot(gs[2, 1])       # Reaction time
        ax7 = fig.add_subplot(gs[2, 2])       # ITI histogram
        ax5 = fig.add_subplot(gs[3, :])       # Raster
                
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
        ax0.text(0, 0.5, summary_text, fontsize=8, va='center', ha='left', family='monospace')
        
        # --- PLOT 1: First poke side by outcome ---
        plot_probability_right_reward_S4(df, ax1)

        # --- PLOT 2: Trial progression over time ---
        plot_trial_progression(ax3, df)

        # --- PLOT 3: Motor and Reaction time ---
        plot_latency_to_first_poke(df, ax4)

        # --- PLOT 5: lick raster and state machine ---
        plot_lick_raster_with_states_S3_S4(ax5, df, fig)
        
        # --- PLOT 6: PC right choice vs P(getting reward for the right side)  ---
        plot_psychometric_curve(df, ax6)

        # --- PLOT 7: histigram iti duration in the session  ---
        plot_iti_histogram(ax7, df)

        plt.subplots_adjust(hspace=0.6, wspace=0.2, top=0.95, bottom=0.09, left= 0.05)
        return fig
        
        #send_slack_plots()