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

# ----------------------------SESSION REPORT PLOTTING FUNCTIONS-----------------------------------

def assign_ports(df: pd.DataFrame) -> pd.DataFrame:
    """Robustly assign left/right/centre poke ports based on system_name."""
    
    system_name = df['system_name'].iloc[0]

    # Definizione delle mappe per ogni system_name
    port_map = {
        9: {
            'left_poke_in': 'Port2In',
            'left_poke_out': 'Port2Out',
            'centre_poke_in': 'Port3In',
            'centre_poke_out': 'Port3Out',
            'right_poke_in': 'Port5In',
            'right_poke_out': 'Port5Out',
        },
        12: {
            'left_poke_in': 'Port7In',
            'left_poke_out': 'Port7Out',
            'centre_poke_in': 'Port4In',
            'centre_poke_out': 'Port4Out',
            'right_poke_in': 'Port1In',
            'right_poke_out': 'Port1Out',
        }
    }

    if system_name not in port_map:
        raise ValueError(f"Unsupported system_name: {system_name}")
    
    mapping = port_map[system_name]

    for new_col, source_col in mapping.items():
        if source_col in df.columns:
            df[new_col] = df[source_col]
        else:
            print(f"[assign_ports] Warning: column '{source_col}' not found in DataFrame. Skipping '{new_col}'.")

    return df



def extract_first_float(val):
            if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                try:
                    return float(ast.literal_eval(val)[0])
                except Exception:
                    return None
            try:
                return float(val) 
            except:
                return None
    
def extract_first_from_list_string(val):
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val) 
                    if isinstance(parsed, list) and len(parsed) > 0:
                        return float(parsed[0])
                except Exception:
                    return None
            return None

def parse_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and compute all S1/S2-related variables from raw trial dataframe."""

    # Basic durations
    df['trial_duration'] = df['TRIAL_END'] - df['TRIAL_START']
    df['sum_s_trial_duration'] = df['trial_duration'].sum()
    df['session_duration'] = df['sum_s_trial_duration'].iloc[0] / 60

    # Time parsing
    for col in [
        'STATE_drink_delay_START', 'STATE_drink_delay_END',
        'STATE_led_on_START', 'STATE_led_on_END',
        'STATE_water_delivery_START', 'STATE_water_delivery_END'
    ]:
        if col in df:
            df[col] = df[col].apply(extract_first_float)
        else:
            print(f"[parse_data] Warning: column '{col}' not found, skipping.")

    # Durations and latencies
    df['duration_drink_delay'] = df.get('STATE_drink_delay_END', 0) - df.get('STATE_drink_delay_START', 0)
    df['duration_led_on'] = df.get('STATE_led_on_END', 0) - df.get('STATE_led_on_START', 0)
    df['reaction_time'] = df.get('STATE_led_on_END', 0) - df.get('STATE_led_on_START', 0)

    # First responses
    df['first_response_right'] = df['right_poke_in'].apply(extract_first_from_list_string) if 'right_poke_in' in df else None
    df['first_response_left'] = df['left_poke_in'].apply(extract_first_from_list_string) if 'left_poke_in' in df else None

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

    # Outcome
    if 'side' in df:
        df["correct_outcome_bool"] = df["first_trial_response"] == df["side"]
        df['true_count'] = df['correct_outcome_bool'].value_counts().get(True, 0)
        df["correct_outcome"] = np.where(df["correct_outcome_bool"], "correct", "incorrect")
        df["correct_outcome_int"] = np.where(df["correct_outcome_bool"], 1, 0)
    else:
        print("[parse_data] Warning: 'side' column not found, skipping outcome computation.")
        df["correct_outcome_bool"] = False
        df["correct_outcome"] = "unknown"
        df["correct_outcome_int"] = 0
        df['true_count'] = 0

    # Summary stats
    df['reaction_time_median'] = df['reaction_time'].median()
    df['tot_correct_choices'] = df['correct_outcome_int'].sum()
    df['right_choices'] = (df['side'] == 'right').sum() if 'side' in df else 0
    df['left_choices'] = (df['side'] == 'left').sum() if 'side' in df else 0

    return df



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


def parse_licks(val):
    if isinstance(val, str):
        try:
            out = ast.literal_eval(val)
            return out if isinstance(out, list) else []
        except:
            return []
    elif isinstance(val, list):
        return val
    return []

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
                                    color='lightblue', alpha=0.3, zorder=1)
                    ax.fill_betweenx([trial - 0.4, trial + 0.4], drink_start, drink_end,
                                    color='orange', alpha=0.3, zorder=1)
                    ax.fill_betweenx([trial - 0.4, trial + 0.4], reward_start, reward_end,
                                    color='lightgreen', alpha=0.3, zorder=1)

                except Exception as e:
                    print(f"Errore al trial {i}: {e}")
                    continue

            # --- PLOT LICKS ---
            ax.scatter(left_raster_times, left_raster_trials, marker='|', color='green',s=60, alpha=1.0, linewidths=0.5, label='Left lick', zorder=10)
            ax.scatter(right_raster_times, right_raster_trials, marker='|', color='purple', s=60,  alpha=1.0, linewidths=0.5, label='Right lick',  zorder=10)

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


def parse_S3_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and compute all S3-related variables from raw trial dataframe."""
    # Basic durations
    df['trial_duration'] = df['TRIAL_END'] - df['TRIAL_START']
    df['sum_s_trial_duration'] = df['trial_duration'].sum()
    df['session_duration'] = df['sum_s_trial_duration'].iloc[0] / 60

    # Time parsing
    df['STATE_drink_delay_START'] = df['STATE_drink_delay_START'].apply(extract_first_float)
    df['STATE_drink_delay_END'] = df['STATE_drink_delay_END'].apply(extract_first_float)
    df['STATE_c_led_on_START'] = df['STATE_c_led_on_START'].apply(extract_first_float)
    df['STATE_c_led_on_END'] = df['STATE_c_led_on_END'].apply(extract_first_float)
    df['STATE_side_led_on_START'] = df['STATE_side_led_on_START'].apply(extract_first_float)
    df['STATE_side_led_on_END'] = df['STATE_side_led_on_END'].apply(extract_first_float)
    df['STATE_water_delivery_START'] = df['STATE_water_delivery_START'].apply(extract_first_float)
    df['STATE_water_delivery_END'] = df['STATE_water_delivery_END'].apply(extract_first_float)
    df['STATE_penalty_START'] = df['STATE_penalty_START'].apply(extract_first_float)
    df['STATE_penalty_END'] = df['STATE_penalty_END'].apply(extract_first_float)

    # Durations and latencies
    df['duration_drink_delay'] = df['STATE_drink_delay_END'] - df['STATE_drink_delay_START']
    df['motor_time'] = df['STATE_side_led_on_END'] - df['STATE_side_led_on_START']
    df['reaction_time'] = df['STATE_c_led_on_END'] - df['STATE_c_led_on_START']  # or water_delivery if appropriate

    # First responses
    df['first_response_right'] = df['right_poke_in'].apply(extract_first_from_list_string)
    df['first_response_left'] = df['left_poke_in'].apply(extract_first_from_list_string)
    df['first_response_center'] = df['centre_poke_in'].apply(extract_first_from_list_string)

    conditions = [
        df['first_response_left'].isna() & df['first_response_right'].isna(),
        df['first_response_left'].isna(),
        df['first_response_right'].isna(),
        df['first_response_left'] <= df['first_response_right'],
        df['first_response_left'] > df['first_response_right'],
    ]
    choices = ["no_response", "right", "left", "left", "right"]
    df["first_trial_response"] = np.select(conditions, choices)

    # Outcome
    df["correct_outcome_bool"] = df["first_trial_response"] == df["side"]
    df['true_count'] = df['correct_outcome_bool'].value_counts().get(True, 0)
    df["correct_outcome"] = np.where(df["first_trial_response"] == df["side"], "correct", "incorrect")
    df["correct_outcome_int"] = np.where(df["first_trial_response"] == df["side"], 1, 0)

    # Summary stats
    df['reaction_time_median'] = df['reaction_time'].median()
    df['tot_correct_choices'] = df['correct_outcome_int'].sum()
    df['right_choices'] = (df['side'] == 'right').sum()
    df['left_choices'] = (df['side'] == 'left').sum()
    return df

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
            ax.plot(i, -0.15 if markersize == 15 else -0.35, '|', color='green', markersize=markersize)
        if row["first_resp_right"]:
            markersize = 15 if row["correct_outcome_int"] == 1 else 5
            ax.plot(i, 1.15 if markersize == 15 else 1.35, '|', color='purple', markersize=markersize)

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

def parse_S4_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and compute all S4-related variables from raw trial dataframe."""
    # Basic durations
    df['trial_duration'] = df['TRIAL_END'] - df['TRIAL_START']
    df['sum_s_trial_duration'] = df['trial_duration'].sum()
    df['session_duration'] = df['sum_s_trial_duration'].iloc[0] / 60

    # Time parsing
    df['STATE_drink_delay_START'] = df['STATE_drink_delay_START'].apply(extract_first_float)
    df['STATE_drink_delay_END'] = df['STATE_drink_delay_END'].apply(extract_first_float)
    df['STATE_c_led_on_START'] = df['STATE_c_led_on_START'].apply(extract_first_float)
    df['STATE_c_led_on_END'] = df['STATE_c_led_on_END'].apply(extract_first_float)
    df['STATE_side_led_on_START'] = df['STATE_side_led_on_START'].apply(extract_first_float)
    df['STATE_side_led_on_END'] = df['STATE_side_led_on_END'].apply(extract_first_float)
    df['STATE_water_delivery_START'] = df['STATE_water_delivery_START'].apply(extract_first_float)
    df['STATE_water_delivery_END'] = df['STATE_water_delivery_END'].apply(extract_first_float)
    df['STATE_penalty_START'] = df['STATE_penalty_START'].apply(extract_first_float)
    df['STATE_penalty_END'] = df['STATE_penalty_END'].apply(extract_first_float)

    # Durations and latencies
    df['duration_drink_delay'] = df['STATE_drink_delay_END'] - df['STATE_drink_delay_START']
    df['motor_time'] = df['STATE_side_led_on_END'] - df['STATE_side_led_on_START']
    df['reaction_time'] = df['STATE_c_led_on_END'] - df['STATE_c_led_on_START']  # or water_delivery if appropriate

    # First responses
    df['first_response_right'] = df['right_poke_in'].apply(extract_first_from_list_string)
    df['first_response_left'] = df['left_poke_in'].apply(extract_first_from_list_string)
    df['first_response_center'] = df['centre_poke_in'].apply(extract_first_from_list_string)

    conditions = [
        df['first_response_left'].isna() & df['first_response_right'].isna(),
        df['first_response_left'].isna(),
        df['first_response_right'].isna(),
        df['first_response_left'] <= df['first_response_right'],
        df['first_response_left'] > df['first_response_right'],
    ]
    choices = ["no_response", "right", "left", "left", "right"]
    df["first_trial_response"] = np.select(conditions, choices)

    # Outcome
    df["correct_outcome_bool"] = df["first_trial_response"] == df["side"]
    df['true_count'] = df['correct_outcome_bool'].value_counts().get(True, 0)
    df["correct_outcome"] = np.where(df["first_trial_response"] == df["side"], "correct", "incorrect")
    df["correct_outcome_int"] = np.where(df["first_trial_response"] == df["side"], 1, 0)

    # Summary stats
    df['reaction_time_median'] = df['reaction_time'].median()
    df['tot_correct_choices'] = df['correct_outcome_int'].sum()
    df['right_choices'] = (df['side'] == 'right').sum()
    df['left_choices'] = (df['side'] == 'left').sum()
    return df

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
        if row["first_trial_response"] == "left":
            ax.plot(row["trial"], -0.15 if correct else -0.35, '|', color="green", markersize=15 if correct else 5)
        elif row["first_trial_response"] == "right":
            ax.plot(row["trial"], 1.15 if correct else 1.35, '|', color="purple", markersize=15 if correct else 5)

    # Draw block probability bars
    if "Block_index" in df.columns:
        unique_blocks = df["Block_index"].unique()
        for block in unique_blocks:
            block_data = df[df["Block_index"] == block]
            start, end = block_data["trial"].min(), block_data["trial"].max()
            block_prob = block_data["probability_r"].iloc[0]
            color = "purple" if block_prob > 0.5 else "green" if block_prob < 0.5 else "blue"
            ax.hlines(y=1.5, xmin=start, xmax=end, colors=color, linewidth=10)
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

def probit(x, beta, alpha):
        # Probit function to generate the curve for the PC
        return 0.5 * (1 + erf((beta * x + alpha) / np.sqrt(2)))
    
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