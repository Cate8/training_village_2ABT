import numpy as np
import pandas as pd
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

# ----------------------------SESSION REPORT PLOTTING FUNCTIONS-----------------------------------

def assign_ports(df: pd.DataFrame) -> pd.DataFrame:
    """Assign left/right poke ports based on system_name."""
    system_name = df['system_name'].iloc[0]

    if system_name == 9:
        df['left_poke_in'] = df['Port2In']
        df['left_poke_out'] = df['Port2Out']
        df['right_poke_in'] = df['Port5In']
        df['right_poke_out'] = df['Port5Out']
    elif system_name == 12:
        df['left_poke_in'] = df['Port7In']
        df['left_poke_out'] = df['Port7Out']
        df['right_poke_in'] = df['Port1In']
        df['right_poke_out'] = df['Port1Out']
    else:
        raise ValueError(f"Unsupported system_name: {system_name}")
    
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
    """Parse and compute all S1- S2-related variables from raw trial dataframe."""

    # Basic durations
    df['trial_duration'] = df['TRIAL_END'] - df['TRIAL_START']
    df['sum_s_trial_duration'] = df['trial_duration'].sum()
    df['session_duration'] = df['sum_s_trial_duration'].iloc[0] / 60

    # Time parsing
    df['STATE_drink_delay_START'] = df['STATE_drink_delay_START'].apply(extract_first_float)
    df['STATE_drink_delay_END'] = df['STATE_drink_delay_END'].apply(extract_first_float)
    df['STATE_led_on_START'] = df['STATE_led_on_START'].apply(extract_first_float)
    df['STATE_led_on_END'] = df['STATE_led_on_END'].apply(extract_first_float)
    df['STATE_water_delivery_START'] = df['STATE_water_delivery_START'].apply(extract_first_float)
    df['STATE_water_delivery_END'] = df['STATE_water_delivery_END'].apply(extract_first_float)

    # Durations and latencies
    df['duration_drink_delay'] = df['STATE_drink_delay_END'] - df['STATE_drink_delay_START']
    df['duration_led_on'] = df['STATE_led_on_END'] - df['STATE_led_on_START']
    df['reaction_time'] = df['STATE_led_on_END'] - df['STATE_led_on_START']  # or water_delivery if appropriate

    # First responses
    df['first_response_right'] = df['right_poke_in'].apply(extract_first_from_list_string)
    df['first_response_left'] = df['left_poke_in'].apply(extract_first_from_list_string)

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

def plot_trial_progression(ax, df):
    """Plot trial progression over time on the given axis."""
    ax.plot(df["TRIAL_START"], df["trial"], label="Trial")
    ax.scatter(df["TRIAL_START"], df["trial"], color="black", s=20)
    ax.set_title("Trial progression over time")
    ax.set_xlabel("TRIAL_START")
    ax.set_ylabel("Trial number")

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