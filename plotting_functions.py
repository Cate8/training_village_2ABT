# ---------------------------- IMPORTS------------------------------------------------------------
import numpy as np
import pandas as pd
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from scipy.special import erf
from scipy.optimize import curve_fit
from matplotlib.patches import Patch
from session_parsing_functions import *

# ----------------------------SESSION REPORT PLOTTING FUNCTIONS-----------------------------------

#PLOT used to represent S1 and S2 sessions
import matplotlib.pyplot as plt
from matplotlib import gridspec

def setup_figure_grid_S1_S2(ncols=3, nrows=3, height_ratios=[0.1, 1, 1], width_ratios=[1, 1, 0.3], figsize=(10, 8)):
    """
    Create a matplotlib figure with a configured GridSpec and font settings.

    Args:
        ncols (int): Number of columns in the grid.
        nrows (int): Number of rows in the grid.
        height_ratios (list): List of relative heights for each row.
        width_ratios (list): List of relative widths for each column.
        figsize (tuple): Size of the figure in inches (width, height).

    Returns:
        fig (Figure): The matplotlib Figure object.
        gs (GridSpec): The GridSpec object to place subplots.
    """
    # Update global font settings
    plt.rcParams.update({'font.size': 6, 'font.family': 'monospace'})

    # Create the figure
    fig = plt.figure(figsize=figsize)

    # Create a GridSpec layout with the specified rows, columns, and ratios
    gs = gridspec.GridSpec(
        nrows=nrows,
        ncols=ncols,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios
    )

    return fig, gs

def plot_session_summary(ax, df):
    """
    Display a text summary of the session on the given matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): The axis where the summary will be plotted.
        df (pd.DataFrame): DataFrame containing session data.

    Returns:
        None
    """
    # Compute session stats
    n_trials = len(df)
    n_correct = df['correct_outcome_int'].sum()
    pct_correct = round(n_correct / n_trials * 100, 2)
    n_left = (df['first_trial_response'] == 'left').sum()
    n_right = (df['first_trial_response'] == 'right').sum()
    n_omit = (df['first_trial_response'] == 'no_response').sum()
    rt_median = round(df['reaction_time'].median(), 2)
    session_duration_min = round(df['session_duration'].iloc[0], 1)

    # Create the summary string
    summary_text = (
        f"Total trials: {n_trials} | Session: {session_duration_min} min | "
        f"Correct: {n_correct} ({pct_correct}%) | Left: {n_left} | Right: {n_right} | "
        f"Omissions: {n_omit} | Median RT: {rt_median} s"
    )

    # Disable axis and display the text
    ax.axis("off")
    ax.text(0, 0.5, summary_text, fontsize=8, va='center', ha='left', family='monospace')

def plot_first_poke_side(ax, df):
    """Plot first poke side by outcome on the given axis."""
    response_map = {"left": -1, "right": 1, "no_response": 0}
    df["first_trial_response_num"] = df["first_trial_response"].map(response_map)

    df['outcome_labels'] = np.where(df['correct_outcome_int'] == 1, 'correct',
                                    np.where(df['correct_outcome_int'].isna(), 'unknown', 'incorrect'))

    sns.scatterplot(
        data=df,
        x="trial",
        y="first_trial_response_num",
        hue="outcome_labels",
        palette={"correct": "black", "incorrect": "red", "unknown": "gray"},
        s=50,
        ax=ax,
    )
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Left", "omission", "Right"])
    ax.set_title("First poke (side)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("First response (side)")
    ax.legend(title="Outcome")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)   

def plot_lick_raster_with_states(ax, df, fig):
            """
            Plot a lick raster aligned to TRIAL_START, with state phases as colored bands.
            Green = left licks, Purple = right licks.
            Lightblue = LED ON, Orange = Drink Delay, Lightgreen = Water Delivery.
            """

            left_raster_trials = []
            left_raster_times = []
            right_raster_trials = []
            right_raster_times = []

            for i, row in df.iterrows():
                try:
                    trial = row['trial']
                    t0 = row['TRIAL_START']

                    # --- LICK TIME  ---
                    left_licks = [lick - t0 for lick in parse_licks(row['left_poke_in'])]
                    right_licks = [lick - t0 for lick in parse_licks(row['right_poke_in'])]

                    left_raster_trials.extend([trial] * len(left_licks))
                    left_raster_times.extend(left_licks)
                    right_raster_trials.extend([trial] * len(right_licks))
                    right_raster_times.extend(right_licks)

                    # --- PHASES (relative to TRIAL_START) ---
                    led_on_start = row['STATE_led_on_START'] - t0
                    led_on_end = row['STATE_led_on_END'] - t0
                    drink_start = row['STATE_drink_delay_START'] - t0
                    drink_end = row['STATE_drink_delay_END'] - t0
                    reward_start = row['STATE_water_delivery_START'] - t0
                    reward_end = row['STATE_water_delivery_END'] - t0

                    # --- COLORED BANDS PER TRIAL ---
                    ax.fill_betweenx([trial - 0.4, trial + 0.4], led_on_start, led_on_end,
                                    color='green', alpha=0.3, zorder=1)
                    ax.fill_betweenx([trial - 0.4, trial + 0.4], drink_start, drink_end,
                                    color='orange', alpha=0.3, zorder=1)
                    ax.fill_betweenx([trial - 0.4, trial + 0.4], reward_start, reward_end,
                                    color='blue', alpha=0.3, zorder=1)

                except Exception as e:
                    print(f"Errore al trial {i}: {e}")
                    continue

            # --- PLOT LICKS ---
            ax.scatter(left_raster_times, left_raster_trials, marker='|', color='green',s=60, alpha=1.0, linewidths=0.5, label='Left lick', zorder=10)
            ax.scatter(right_raster_times, right_raster_trials, marker='|', color='purple', s=60,  alpha=1.0, linewidths=0.5, label='Right lick',  zorder=10)
            
            # Legend for states
            state_legend = [
                Patch(facecolor='green', alpha=0.3, label='side led ON'),
                Patch(facecolor='orange', alpha=0.3, label='ITI'),
                Patch(facecolor='blue', alpha=0.3, label='Reward')
            ]

            # Legend for licks
            lick_legend = [
                Patch(color='green', label='Left lick'),
                Patch(color='purple', label='Right lick')
            ]

            # Combine and place legend outside bottom-right
            ax.legend(handles=state_legend + lick_legend,
                    loc='lower right',
                    bbox_to_anchor=(1.75, -0.1),
                    fontsize=6,
                    frameon=False)

            ax.set_title("Lick raster (aligned to trial start)")
            ax.set_xlabel("Time from trial start (s)")
            ax.set_ylabel("Trial")
            ax.set_xlim(left=0)
            ax.set_ylim(df['trial'].min() - 1, df['trial'].max() + 1)
            ax.invert_yaxis()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

def plot_trial_progression(ax, df):
    """Plot trial progression over time on the given axis."""
    df['min_from_session_start'] = (df['TRIAL_START'] - df['TRIAL_START'].iloc[0]) / 60
    ax.plot(df["min_from_session_start"], df["trial"], label="Trial")
    ax.scatter(df["min_from_session_start"], df["trial"], color="black", s=20)
    ax.set_title("Trial progression over time")
    ax.set_xlabel("mins from session start")
    ax.set_ylabel("Trial number")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_rolling_accuracy(ax, df, window=5):
    """Plot rolling accuracy (%) over trials on the given axis."""
    df['rolling_accuracy'] = df['correct_outcome_int'].rolling(window=window, min_periods=1).mean() * 100

    ax.plot(df['trial'], df['rolling_accuracy'], color='blueviolet', linestyle='-',
            linewidth=2, marker='o', markersize=4)
    ax.axhline(y=50, color='black', linestyle='--')
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Trial")
    ax.set_ylabel("%")
    ax.set_title("Trials accuracy (rolling average)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_reaction_time(ax, df):
    """Plot reaction time (RT) over trials on a log scale."""
    ax.plot(df['trial'], df['reaction_time'], color='dodgerblue', linewidth=2, markersize=8)

    # Log scale for y-axis
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

    # Titles and labels
    ax.set_title('Reaction time (RT)')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Latency to poke (s)')

    # Horizontal grid lines at specific tick values
    custom_yticks = [1, 10, 100]
    for y in custom_yticks:
        ax.axhline(y=y, linestyle='--', color='lightgray', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


#PLOT used to represent S3 and S4 sessions
def plot_right_reward_probability(df, ax=None):
    """Plot the probability of right reward over trials."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    df = df.replace(np.nan, 0)

    df['rolling_prob'] = df['correct_outcome_int'].rolling(window=5, min_periods=1).mean()
    df['right_rewards'] = ((df['side'] == 'right') & (df['correct_outcome_int'] == 1)).astype(int)
    df['rolling_avg_right_reward'] = df["right_rewards"].rolling(window=5, min_periods=1).mean()

    left = df['first_response_left'].fillna(np.inf)
    right = df['first_response_right'].fillna(np.inf)

    conditions = [
        df['first_response_left'].isna() & df['first_response_right'].isna(),
        df['first_response_left'].isna(),
        df['first_response_right'].isna(),
        left <= right,
        left > right,
    ]
    choices = ["no_response", "right", "left", "left", "right"]
    df["first_trial_response"] = np.select(conditions, choices)

    df["first_resp_left"] = (df["first_trial_response"] == "left").astype(int)
    df["first_resp_right"] = (df["first_trial_response"] == "right").astype(int)

    # --- Plot rolling prob curve ---
    ax.plot(df["trial"], df["rolling_avg_right_reward"], color='mediumturquoise', linewidth=1, label='Rolling P(right reward)', linestyle='-')
    ax.plot(df["trial"], df["rolling_prob"], color='black', linewidth=1, linestyle='--', label= 'Rolling accuracy')
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1])
    ax.axhline(0.5, linestyle='--', color='lightgray', alpha=0.7)
    ax.axhline(0, linestyle='solid', color='black', alpha=0.7)
    ax.axhline(1, linestyle='solid', color='black', alpha=0.7)

    # --- Plot ticks ---
    for i, row in df.iterrows():
        if row["first_resp_left"]:
            markersize = 15 if row["correct_outcome_int"] == 1 else 5
            ax.plot(i, 1.15 if markersize == 15 else 1.35, '|', color='purple', markersize=markersize)
        if row["first_resp_right"]:
            markersize = 15 if row["correct_outcome_int"] == 1 else 5
            ax.plot(i, -0.15 if markersize == 15 else -0.35, '|', color='green', markersize=markersize)

    # --- SIDE LABLES ---
    ax.text(1.02, 0.1, 'L', ha='left', va='top', color='green', transform=ax.transAxes, fontsize=10)
    ax.text(1.02, 0.9, 'R', ha='left', va='bottom', color='purple', transform=ax.transAxes, fontsize=10)
    ax.text(1.02, 0.455, 'C', ha='left', va='bottom', color='black', transform=ax.transAxes, fontsize=10)

    #----- legend -----
    ax.legend(loc='lower right', fontsize=7, frameon=False)
    # --- x axis ---
    selected_trials = df["trial"][::19]
    ax.set_xticks(selected_trials)
    ax.set_xticklabels(selected_trials)
    ax.set_xlabel("Trial")
    ax.set_ylabel("P(right)")
    ax.set_title("Rolling accuracy for right-side rewards")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

def plot_latency_to_first_poke(df, ax=None):
    """
    Plot latency to the first correct poke (side and centre) with log y-scale.

    Parameters:
    - df: pandas DataFrame, deve contenere le colonne 'trial', 'side_response_latency', 'centre_response_latency'
    - ax: matplotlib axis object, opzionale. Se None, ne crea uno.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df.trial, df.motor_time, color='dodgerblue', label='MT', linewidth=1)
    ax.plot(df.trial, df.reaction_time, color='black', label='RT', linewidth=1)

    # Y log scale and ticks 
    custom_yticks = [0.1, 1, 10, 20, 50, 100, 200, 300]
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    for y in custom_yticks:
        ax.axhline(y=y, linestyle='--', color='lightgray', alpha=0.7)

    ax.set_title('Latency to first correct poke')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Latency (s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1, -0.2), ncol=2, frameon=False)

    return ax

def plot_lick_raster_with_states_S3(ax, df, fig):
            """
            Plot a lick raster aligned to TRIAL_START, with state phases as colored bands.
            Green = left licks, Purple = right licks.
            Lightblue = LED ON, Orange = Drink Delay, Lightgreen = Water Delivery.
            """

            left_raster_trials = []
            left_raster_times = []
            centre_raster_trials = []
            centre_raster_times = []
            right_raster_trials = []
            right_raster_times = []

            for i, row in df.iterrows():
                try:
                    trial = row['trial']
                    t0 = row['TRIAL_START']

                    # --- LICK TIME  ---
                    left_licks = [lick - t0 for lick in parse_licks(row['left_poke_in'])]
                    centre_licks = [lick - t0 for lick in parse_licks(row['centre_poke_in'])]
                    right_licks = [lick - t0 for lick in parse_licks(row['right_poke_in'])]

                    left_raster_trials.extend([trial] * len(left_licks))
                    left_raster_times.extend(left_licks)
                    centre_raster_trials.extend([trial] * len(centre_licks))
                    centre_raster_times.extend(centre_licks)
                    right_raster_trials.extend([trial] * len(right_licks))
                    right_raster_times.extend(right_licks)

                    # --- PHASES (relative to TRIAL_START) ---
                    c_led_on_start = row['STATE_c_led_on_START'] - t0
                    c_led_on_end = row['STATE_c_led_on_END'] - t0
                    side_led_on_start = row['STATE_side_led_on_START'] - t0
                    side_led_on_end = row['STATE_side_led_on_END'] - t0
                    drink_start = row['STATE_drink_delay_START'] - t0
                    drink_end = row['STATE_drink_delay_END'] - t0
                    reward_start = row['STATE_water_delivery_START'] - t0
                    reward_end = row['STATE_water_delivery_END'] - t0
                    penalty_start = row['STATE_water_delivery_START'] - t0
                    penalty_end = row['STATE_water_delivery_END'] - t0

                    # --- COLORED BANDS PER TRIAL ---
                    ax.fill_betweenx([trial - 0.4, trial + 0.4], c_led_on_start, c_led_on_end,
                                    color='yellow', alpha=0.3, zorder=1)
                    ax.fill_betweenx([trial - 0.4, trial + 0.4], side_led_on_start, side_led_on_end,
                                    color='lightblue', alpha=0.3, zorder=1)
                    ax.fill_betweenx([trial - 0.4, trial + 0.4], drink_start, drink_end,
                                    color='orange', alpha=0.3, zorder=1)
                    ax.fill_betweenx([trial - 0.4, trial + 0.4], penalty_start, penalty_end,
                                    color='red', alpha=0.3, zorder=1)
                    ax.fill_betweenx([trial - 0.4, trial + 0.4], reward_start, reward_end,
                                    color='lightgreen', alpha=0.3, zorder=1)

                except Exception as e:
                    print(f"Errore al trial {i}: {e}")
                    continue

            # --- PLOT LICKS ---
            ax.scatter(left_raster_times, left_raster_trials, marker='|', color='green',s=60, alpha=1.0, linewidths=0.5, label='Left lick', zorder=10)
            ax.scatter(right_raster_times, right_raster_trials, marker='|', color='purple', s=60,  alpha=1.0, linewidths=0.5, label='Right lick',  zorder=10)
            ax.scatter(centre_raster_times, centre_raster_trials, marker='|', color='black', s=60,  alpha=1.0, linewidths=0.5, label='Right lick',  zorder=10)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_title("Lick raster (aligned to trial start)")
            ax.set_xlabel("Time from trial start (s)")
            ax.set_ylabel("Trial")
            ax.set_xlim(left=0)
            ax.set_ylim(df['trial'].min() - 1, df['trial'].max() + 1)
            ax.invert_yaxis()


def plot_probability_right_reward_S4(df: pd.DataFrame, ax=None) -> plt.Axes:
    """
    Plot probability of right reward vs. actual right choices.
    Shows rolling average of right choices, expected probabilities,
    response ticks, and block-level structure.

    Parameters:
    - df: DataFrame with trial data.
    - ax: matplotlib axis to plot on (optional).

    Returns:
    - ax: matplotlib axis with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Clean & prepare
    df = df.copy()
    df = df.replace(np.nan, 0)

    # Rolling average of right choices
    df["right_choice"] = (df["first_trial_response"] == "right").astype(int)
    df["rolling_avg_right"] = df["right_choice"].rolling(window=5, min_periods=1).mean()

    # Plot expected probability of reward
    ax.plot(df["trial"], df["probability_r"], label=" P(reward on right)",
            color="black", linewidth=1, alpha=0.7)

    # Plot actual right choices rolling average
    ax.plot(df["trial"], df["rolling_avg_right"], label="Right choice frequency",
            color="mediumturquoise", linewidth=2)

    # Plot response ticks (green = left, purple = right)
    for i, row in df.iterrows():
        correct = row["correct_outcome_int"] == 1
        if row["first_trial_response"] == "right":
            ax.plot(row["trial"], -0.15 if correct else -0.35, '|', color="green", markersize=15 if correct else 5)
        elif row["first_trial_response"] == "left":
            ax.plot(row["trial"], 1.15 if correct else 1.35, '|', color="purple", markersize=15 if correct else 5)


    # Draw block probability bars
    if "Block_index" in df.columns:
        unique_blocks = df["Block_index"].unique()
        for block in unique_blocks:
            block_data = df[df["Block_index"] == block]
            start, end = block_data["trial"].min(), block_data["trial"].max()
            block_prob = block_data["probability_r"].iloc[0]
            color = "purple" if block_prob > 0.5 else "green" if block_prob < 0.5 else "blue"
            ax.hlines(y=1.7, xmin=start, xmax=end, colors=color, linewidth=10)
            ax.text((start + end) / 2, 1.6, f"{block_prob:.2f}", ha="center", va="center",
                    fontsize=6, backgroundcolor="white")

        # Vertical lines for block changes
        changes = df["Block_index"].diff().fillna(0).ne(0)
        for trial in df[changes]["trial"]:
            ax.axvline(x=trial, color="gray", linestyle="--")

    # Axes formatting
    ax.set_ylim(-0.5, 1.7)
    ax.set_yticks([0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1])
    ax.axhline(y=0.5, linestyle="--", color="lightgray", alpha=0.7)
    ax.axhline(y=0, linestyle="solid", color="black", alpha=0.7)
    ax.axhline(y=1, linestyle="solid", color="black", alpha=0.7)

    # Labels
    ax.text(1.02, 0.1, "L", transform=ax.transAxes, color="green", fontsize=10)
    ax.text(1.02, 0.9, "R", transform=ax.transAxes, color="purple", fontsize=10)
    ax.text(1.02, 0.46, "C", transform=ax.transAxes, color="black", fontsize=10)

    # Title and axis
    ax.set_title("P(reward on right) vs. Choice (Rolling)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("P(right)")
    ax.set_xticks(df["trial"][::20])
    ax.set_xticklabels(df["trial"][::20])

    # Legend
    ax.legend(loc="upper left")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax

def plot_psychometric_curve(df, ax=None):
    """Plot psychometric curve: proportion of right choices vs. right reward probability."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    # Ensure columns are float and clean
    df = df.copy()
    df['probability_r'] = df['probability_r'].astype(float)
    df['first_trial_response_num'] = df['first_trial_response'].apply(lambda x: 1 if x == 'right' else 0)

    # Compute right choice rate per unique prob
    probs = np.sort(df['probability_r'].unique())
    right_choice_freq = [
        df[df['probability_r'] == p]['first_trial_response_num'].mean()
        for p in probs
    ]

    # Fit probit curve
    try:
        pars, _ = curve_fit(probit, df['probability_r'], df['first_trial_response_num'], p0=[0, 1])
    except RuntimeError:
        pars = [0, 0]  # fallback if fitting fails

    # Plot data points
    ax.scatter(probs, right_choice_freq, color='indianred', s=20, label='Data')

    # Plot fitted curve
    x = np.linspace(0, 1, 100)
    ax.plot(x, probit(x, *pars), color='indianred', linewidth=2, label='Probit Fit')

    # Style
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', linestyle='--')
    ax.axvline(0.5, color='gray', linestyle='--')
    ax.set_xlabel('Right reward probability')
    ax.set_ylabel('Right choice rate')
    ax.set_title('Psychometric curve')
    ax.legend(loc='lower right', fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax

def plot_lick_raster_with_states_S3_S4(ax, df, fig=None):
    """
    Plot a lick raster aligned to TRIAL_START with trial on x-axis and time on y-axis.
    Green = left licks, Purple = right licks, Gray = center licks.
    Background bands: Yellow = center LED, Lightblue = side LED, Orange = drink delay, Red = penalty, Lightgreen = reward.
    """

    left_raster_trials, left_raster_times = [], []
    centre_raster_trials, centre_raster_times = [], []
    right_raster_trials, right_raster_times = [], []

    for i, row in df.iterrows():
        try:
            trial = row['trial']
            t0 = row['TRIAL_START']

            # Lick times relative to trial start
            left_licks = [lick - t0 for lick in parse_licks(row.get('left_poke_in', ''))]
            centre_licks = [lick - t0 for lick in parse_licks(row.get('centre_poke_in', ''))]
            right_licks = [lick - t0 for lick in parse_licks(row.get('right_poke_in', ''))]

            left_raster_trials.extend([trial] * len(left_licks))
            left_raster_times.extend(left_licks)
            centre_raster_trials.extend([trial] * len(centre_licks))
            centre_raster_times.extend(centre_licks)
            right_raster_trials.extend([trial] * len(right_licks))
            right_raster_times.extend(right_licks)

            # Phases
            def rel(key):
                return row.get(key, t0) - t0

            bands = [
                (rel('STATE_c_led_on_START'), rel('STATE_c_led_on_END'), 'yellow', 'Center LED'),
                (rel('STATE_side_led_on_START'), rel('STATE_side_led_on_END'), 'orchid', 'Side LED'),
                (rel('STATE_drink_delay_START'), rel('STATE_drink_delay_END'), 'orange', 'ITI'),
                (rel('STATE_water_delivery_START'), rel('STATE_water_delivery_END'), 'blue', 'Reward'),
                (rel('STATE_penalty_START'), rel('STATE_penalty_END'), 'firebrick', 'Penalty')
            ]

            for start, end, color, _ in bands:
                ax.fill_between([trial - 0.4, trial + 0.4], start, end, color=color, alpha=0.3, zorder=1)

        except Exception as e:
            print(f"[Raster] Trial {i} skipped due to error: {e}")
            continue

    # Plot licks
    ax.scatter(left_raster_trials, left_raster_times, marker='_', color='green', s=40, alpha=0.7, label='Left')
    ax.scatter(centre_raster_trials, centre_raster_times, marker='_', color='black', s=40, alpha=0.7, label='Centre')
    ax.scatter(right_raster_trials, right_raster_times, marker='_', color='purple', s=40, alpha=0.7, label='Right')

    ax.set_xlabel("Trial")
    ax.set_ylabel("Time from trial start (s)")
    ax.set_title("Lick Raster (aligned to TRIAL_START)")
    ax.set_ylim(0, df['trial_duration'].max() + 1)
    ax.spines[['top', 'right']].set_visible(False)

    # Legend for licks
    lick_legend = [
        Patch(color='green', label='Left'),
        Patch(color='purple', label='Right'),
        Patch(color='black', label='Center')
    ]

    # Legend for states
    state_legend = [
        Patch(facecolor='yellow', alpha=0.5, label='Center LED'),
        Patch(facecolor='orchid', alpha=0.5, label='Side LED'),
        Patch(facecolor='orange', alpha=0.5, label='ITI'),
        Patch(facecolor='blue', alpha=0.5, label='Reward'),
        Patch(facecolor='firebrick', alpha=0.5, label='Penalty')
    ]

    all_handles = lick_legend + state_legend
    ax.legend(handles=all_handles, loc='center left', bbox_to_anchor=(0.99, 0.5),
              fontsize=6, frameon=False)


    return ax
    

def plot_iti_histogram(ax, df, bins=10):
    """
    Plot histogram of ITI durations.
    
    Parameters:
        ax : matplotlib.axes.Axes
            The axis to plot on.
        df : pd.DataFrame
            The dataframe containing the 'iti_duration' column.
        bins : int
            Number of histogram bins.
    """
    if 'iti_duration' not in df.columns:
        ax.text(0.5, 0.5, 'No ITI data', ha='center', va='center', fontsize=8)
        ax.set_axis_off()
        return

    iti_data = df['iti_duration'].dropna()
    
    ax.hist(iti_data, bins=bins, color='lightseagreen', edgecolor='black', alpha=0.8)
    ax.set_title("ITI duration histogram")
    ax.set_xlabel("ITI duration (s)")
    ax.set_ylabel("Number of trials")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)