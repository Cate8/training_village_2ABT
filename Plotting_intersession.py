import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# ------------------------------ Data Processing Functions ------------------------------
def assign_ports_intersession(df: pd.DataFrame) -> pd.DataFrame:
    """Robustly assign left/right/centre poke ports based on system_name, even with weird column names."""

    # Clean column names to avoid issues with hidden spaces or strange characters
    df.columns = df.columns.str.strip()  # remove leading/trailing spaces
    df.columns = df.columns.str.replace('\u200b', '', regex=True)  # remove zero-width spaces

    system_name = df['system_name'].iloc[0]

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
        raise ValueError(f"[assign_ports] Unsupported system_name: {system_name}")

    mapping = port_map[system_name]

    # Build a case-insensitive mapping of existing columns
    df_cols_lower = {col.lower(): col for col in df.columns}

    for new_col, source_col in mapping.items():
        source_col_lower = source_col.lower()
        if source_col_lower in df_cols_lower:
            actual_col = df_cols_lower[source_col_lower]
            df[new_col] = df[actual_col]
            print(f"[assign_ports] Assigned '{new_col}' from column '{actual_col}'")
        else:
            print(f"[assign_ports] WARNING: source column '{source_col}' not found in DataFrame. Skipping '{new_col}'.")

    return df

# ---------------------------- Lick raster S1 and S2 -------------------------------------
def parse_licks(raw):
    """
    Safely parse a string of lick times separated by commas into a list of floats.
    Returns an empty list if input is NaN, None or invalid.
    """
    if pd.isna(raw) or raw is None:
        return []
    if isinstance(raw, (int, float)):  # Already numeric → wrap in list
        return [float(raw)]
    try:
        return [float(val.strip()) for val in str(raw).split(',') if val.strip()]
    except Exception as e:
        print(f"[WARN] Could not parse licks '{raw}': {e}")
        return []


def safe_float(value):
    """Converts value to float, returns np.nan if conversion fails."""
    try:
        if pd.isna(value) or value in ["nan", "None", None, ""]:
            return np.nan
        return float(value)
    except Exception:
        return np.nan

def parse_licks(raw):
    """
    Safely parse a string of lick times separated by commas into a list of floats.
    Returns an empty list if input is NaN, None or invalid.
    """
    if pd.isna(raw) or raw is None:
        return []
    if isinstance(raw, (int, float)):  # Already numeric → wrap in list
        return [float(raw)]
    try:
        return [float(val.strip()) for val in str(raw).split(',') if val.strip()]
    except Exception as e:
        print(f"[WARN] Could not parse licks '{raw}': {e}")
        return []

def plot_lick_raster_with_states(ax, df, fig):
    """
    Plot a lick raster aligned to TRIAL_START, with state phases as colored bands.
    Green = left licks, Purple = right licks.
    Lightblue = LED ON, Orange = Drink Delay, Blue = Water Delivery.
    """

    left_raster_trials = []
    left_raster_times = []
    right_raster_trials = []
    right_raster_times = []

    for i, row in df.iterrows():
        try:
            trial = row['trial']
            t0 = safe_float(row['TRIAL_START'])
            if np.isnan(t0):
                raise ValueError(f"Invalid TRIAL_START at trial {trial}")

            # --- LICK TIMES ---
            left_licks = [
                lick_num - t0
                for lick_num in [safe_float(l) for l in parse_licks(row['left_poke_in'])]
                if not np.isnan(lick_num)
            ]
            right_licks = [
                lick_num - t0
                for lick_num in [safe_float(l) for l in parse_licks(row['right_poke_in'])]
                if not np.isnan(lick_num)
            ]

            left_raster_trials.extend([trial] * len(left_licks))
            left_raster_times.extend(left_licks)
            right_raster_trials.extend([trial] * len(right_licks))
            right_raster_times.extend(right_licks)

            # --- STATES ---
            led_on_start = safe_float(row['STATE_led_on_START']) - t0
            led_on_end   = safe_float(row['STATE_led_on_END'])   - t0
            drink_start  = safe_float(row['STATE_drink_delay_START']) - t0
            drink_end    = safe_float(row['STATE_drink_delay_END'])   - t0
            reward_start = safe_float(row['STATE_water_delivery_START']) - t0
            reward_end   = safe_float(row['STATE_water_delivery_END'])   - t0

            # --- DRAW STATE BANDS ---
            if not np.isnan(led_on_start) and not np.isnan(led_on_end):
                ax.fill_betweenx([trial-0.4, trial+0.4], led_on_start, led_on_end,
                                 color='lightblue', alpha=0.3, zorder=1)
            if not np.isnan(drink_start) and not np.isnan(drink_end):
                ax.fill_betweenx([trial-0.4, trial+0.4], drink_start, drink_end,
                                 color='orange', alpha=0.3, zorder=1)
            if not np.isnan(reward_start) and not np.isnan(reward_end):
                ax.fill_betweenx([trial-0.4, trial+0.4], reward_start, reward_end,
                                 color='blue', alpha=0.3, zorder=1)

        except Exception as e:
            print(f"Error at trial {trial}: {e}")
            continue

    # --- PLOT LICKS ---
    ax.scatter(left_raster_times, left_raster_trials, marker='|', color='green', s=60, alpha=1.0, linewidths=0.5, label='Left lick', zorder=10)
    ax.scatter(right_raster_times, right_raster_trials, marker='|', color='purple', s=60, alpha=1.0, linewidths=0.5, label='Right lick', zorder=10)

    # Legends
    state_legend = [
        Patch(facecolor='lightblue', alpha=0.3, label='LED ON'),
        Patch(facecolor='orange', alpha=0.3, label='Drink Delay'),
        Patch(facecolor='blue', alpha=0.3, label='Water Delivery')
    ]
    lick_legend = [
        Patch(color='green', label='Left lick'),
        Patch(color='purple', label='Right lick')
    ]

    ax.legend(handles=state_legend + lick_legend,
              loc='lower right', bbox_to_anchor=(1.75, -0.1),
              fontsize=6, frameon=False)

    ax.set_title("Lick raster (aligned to trial start)")
    ax.set_xlabel("Time from trial start (s)")
    ax.set_ylabel("Trial")
    ax.set_xlim(left=0)
    ax.set_ylim(df['trial'].min() - 1, df['trial'].max() + 1)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ----------------------- Data Plotting Functions S3 and S4 ------------------------------
# Convert date strings to datetime objects
def aggregate_data(df):
    """
    Convert the 'date' column to pandas datetime, inferring the format automatically.
    """
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # infer format automatically
    return df

# Probit function used to fit psychometric curves
def probit(x, beta, alpha):
    """
    Probit function: computes the cumulative probability using the probit model.
    """
    return 0.5 * (1 + erf((beta * x + alpha) / np.sqrt(2)))

def plot_psychometric_curves(df, axes):
    """
    Plot psychometric curves for the last five days of data.
    Each curve shows the relationship between 'probability_r' and right choice rate,
    fitted with a probit function, with curves separated by task within the same day.

    Args:
        df (pd.DataFrame): The filtered dataframe.
        axes (list): List of matplotlib axes where plots will be drawn.
    """
   
    df['date'] = pd.to_datetime(df['date']).dt.date.astype(str)
    
    # Extract last 5 days 
    last_five_dates = df['date'].dropna().unique()[-5:]


    unique_tasks = sorted(df['task'].dropna().unique())
    task_colors = {task: cm.tab10(i % 10) for i, task in enumerate(unique_tasks)}

    for i, date in enumerate(last_five_dates):
        if i >= len(axes):
            break  
        ax = axes[i]
        
        df_day = df[df['date'] == date].copy()
        tasks_today = df_day['task'].dropna().unique()

        for task in tasks_today:
            group = df_day[df_day['task'] == task].copy()

            group['probability_r'] = group['probability_r'].astype(float)
            group['first_trial_response'] = group['first_trial_response'].apply(lambda x: 1 if x == 'right' else 0)

            probs = np.sort(group['probability_r'].unique())
            right_choice_freq = []
            for prob in probs:
                mask_prob = group['probability_r'] == prob
                n_trials = mask_prob.sum()
                n_right = (mask_prob & (group['first_trial_response'] == 1)).sum()
                right_choice_freq.append(n_right / n_trials if n_trials > 0 else np.nan)

            # Fit probit
            try:
                pars, _ = curve_fit(probit, group['probability_r'], group['first_trial_response'], p0=[0, 1])
                x = np.linspace(0, 1, 100)
                ax.plot(x, probit(x, *pars), label=task, color=task_colors[task], linewidth=2)
            except RuntimeError:
                print(f"[plot_psychometric_curves] Fit failed for {task} on {date}")

            ax.scatter(probs, right_choice_freq, color=task_colors[task], s=20, alpha=0.8)

        ax.set_ylim(0, 1)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle='--')
        ax.axvline(0.5, color='gray', linestyle='--')
        ax.set_title(pd.to_datetime(date).strftime("%Y-%m-%d"), fontsize=15)
        ax.set_xlabel('Probability', fontsize=15)
        ax.set_ylabel('Right Choice Rate' if i == 0 else '', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Imposta ticks uniformi e label leggibili sull'asse X
        ax.set_xticks(np.arange(0.1, 1.0, 0.1))
        ax.set_xticklabels([f"{tick:.1f}" for tick in np.arange(0.1, 1.0, 0.1)], fontsize=13, rotation=45)

        # Etichetta asse X centrata in tutti i plot
        ax.set_xlabel('P(rightward reward)', fontsize=15)

        if i == 0:
            ax.set_ylabel('Right Choice Rate', fontsize=13)
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        

        if len(tasks_today) > 1:
            ax.legend(fontsize=10, loc='best')

def plot_correct_choice_curves(df, axes):
    """
    Plot correct choice rate curves for the last five days of data,
    showing the fraction of correct responses (choosing the port with highest probability).
    Curves are grouped by date (daily) and color-coded by task.
    
    Args:
        df (pd.DataFrame): The filtered dataframe.
        axes (list): List of matplotlib axes where plots will be drawn.
    """

    df['date'] = pd.to_datetime(df['date']).dt.date.astype(str)

    df['probability_l'] = 1 - df['probability_r'].astype(float)
    df['prob_rwd_correct_resp'] = np.where(df['probability_r'] >= 0.5, df['probability_r'], 1 - df['probability_r']).astype(float)

    df['high_choice'] = (
        ((df['first_trial_response'] == 'right') & (df['probability_r'] > df['probability_l'])) |
        ((df['first_trial_response'] == 'left') & (df['probability_l'] > df['probability_r']))
    ).astype(int)

    last_five_dates = df['date'].dropna().unique()[-5:]

    unique_tasks = sorted(df['task'].unique())
    task_colors = {task: plt.cm.tab10(i % 10) for i, task in enumerate(unique_tasks)}

    for i, date in enumerate(last_five_dates):
        if i >= len(axes):
            break  
        ax = axes[i]
        day_df = df[df['date'] == date]

        for task in day_df['task'].unique():
            task_df = day_df[day_df['task'] == task]

            probs = np.sort(task_df['prob_rwd_correct_resp'].unique())
            correct_freq = []

            for prob in probs:
                idx = task_df['prob_rwd_correct_resp'] == prob
                n_trials = np.sum(idx)
                n_correct = np.sum(task_df.loc[idx, 'high_choice'])
                correct_freq.append(n_correct / n_trials if n_trials > 0 else np.nan)

            try:
                pars, _ = curve_fit(probit, task_df['prob_rwd_correct_resp'], task_df['high_choice'], p0=[0, 1])
                x = np.linspace(0, 1, 100)
                ax.plot(x, probit(x, *pars), label=task, color=task_colors[task], linewidth=2)
            except RuntimeError:
                print(f"[plot_correct_choice_curves] Fit failed for {date} task {task}")

            ax.scatter(probs, correct_freq, color=task_colors[task], s=20)

 
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle='--')
        ax.axvline(0.5, color='gray', linestyle='--')
        ax.set_title(f'{date}', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xticks(np.arange(0.1, 1.0, 0.1))
        ax.set_xticklabels([f"{tick:.1f}" for tick in np.arange(0.1, 1.0, 0.1)], fontsize=13, rotation=45)

        ax.set_xlabel('P(rightward reward)', fontsize=15)

        if i == 0:
            ax.set_ylabel('Fraction of correct responses', fontsize=13)
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])

        if len(day_df['task'].unique()) > 1:
            ax.legend(fontsize=8, loc='lower right')

def plot_aggregate_psychometric_right_choice(df, ax):
    """
    Plot an aggregated psychometric curve for right choices over the last 15 days.
    Each point represents the average right choice rate across all trials in the last 15 days
    for each unique probability level.
    
    Args:
        df (pd.DataFrame): The filtered dataframe.
        ax (matplotlib.axes.Axes): The matplotlib axis where the plot will be drawn.
    """
    # Preparazione date e filtro ultimi 15 giorni
    df['date'] = pd.to_datetime(df['date']).dt.date
    last_15_dates = df['date'].dropna().unique()[-15:]
    df_agg = df[df['date'].isin(last_15_dates)].copy()

    if df_agg.empty:
        ax.text(0.5, 0.5, 'No data\n(last 15 days)', ha='center', va='center', fontsize=15)
        ax.set_axis_off()
        return

    df_agg['probability_r'] = df_agg['probability_r'].astype(float)
    df_agg['first_trial_response'] = df_agg['first_trial_response'].apply(lambda x: 1 if x == 'right' else 0)

    # Raggruppa per probability_r → media e SEM sulle risposte right
    grouped = df_agg.groupby('probability_r')['first_trial_response'].agg(['mean', 'std', 'count']).reset_index()
    grouped['sem'] = grouped['std'] / grouped['count']**0.5

    # Fit sulla media dei punti
    try:
        pars, _ = curve_fit(probit, grouped['probability_r'], grouped['mean'], p0=[0, 1])
        x_fit = np.linspace(0, 1, 100)
        ax.plot(x_fit, probit(x_fit, *pars), color='black', linewidth=2)
    except RuntimeError:
        print("[plot_aggregate_psychometric_right_choice] Fit failed on aggregated means.")

    # Scatter: un punto per livello di probabilità
    ax.errorbar(grouped['probability_r'], grouped['mean'],
                yerr=grouped['sem'], fmt='o', color='black', ecolor='gray', capsize=4)

    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', linestyle='--')
    ax.axvline(0.5, color='gray', linestyle='--')
    ax.set_title('Last 15 days', fontsize=12, pad=10)
    ax.set_xlabel('P(rightward reward)', fontsize=15)
    ax.set_ylabel('Fraction of correct responses', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_xticks(np.arange(0.1, 1.0, 0.1))
    ax.set_xticklabels([f"{tick:.1f}" for tick in np.arange(0.1, 1.0, 0.1)], fontsize=13, rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_aggregate_correct_choice_mean(df, ax):
    """
    Plot an aggregated psychometric curve for correct choices from the last 15 days.
    Each point represents the average correct response rate across all trials in the last 15 days
    for each unique probability level.
    
    Args:
        df (pd.DataFrame): DataFrame with trial data.
        ax (matplotlib.axes): Axis to plot on.
    """
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['probability_r'] = df['probability_r'].astype(float)
    df['probability_l'] = 1 - df['probability_r']
    df['prob_rwd_correct_resp'] = np.where(df['probability_r'] >= 0.5, df['probability_r'], 1 - df['probability_r'])

    # Calcola correct responses
    df['high_choice'] = (
        ((df['first_trial_response'] == 'right') & (df['probability_r'] > df['probability_l'])) |
        ((df['first_trial_response'] == 'left') & (df['probability_l'] > df['probability_r']))
    ).astype(int)

    # Ultimi 15 giorni
    last_15_dates = df['date'].dropna().unique()[-15:]
    df_agg = df[df['date'].isin(last_15_dates)].copy()

    # Raggruppa per probabilità → media e SEM sulle risposte corrette (tutti i trial negli ultimi 15 giorni)
    grouped = df_agg.groupby('prob_rwd_correct_resp')['high_choice'].agg(['mean', 'std', 'count']).reset_index()
    grouped['sem'] = grouped['std'] / grouped['count']**0.5

    # Fit sulla media dei punti
    try:
        pars, _ = curve_fit(probit, grouped['prob_rwd_correct_resp'], grouped['mean'], p0=[0, 1])
        x_fit = np.linspace(0, 1, 100)
        ax.plot(x_fit, probit(x_fit, *pars), color='black', linewidth=2)
    except RuntimeError:
        print("[plot_aggregate_correct_choice_mean] Fit failed on aggregated means.")

    # Scatter: un punto per livello di probabilità
    ax.errorbar(grouped['prob_rwd_correct_resp'], grouped['mean'],
                yerr=grouped['sem'], fmt='o', color='black', ecolor='gray', capsize=4)

    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', linestyle='--')
    ax.axvline(0.5, color='gray', linestyle='--')
    ax.set_title('Last 15 days', fontsize=12, pad=10)
    ax.set_xlabel('P(rightward reward)', fontsize=15)
    ax.set_ylabel('Fraction of correct responses', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([0.1 * i for i in range(1, 10)])
    ax.set_xticklabels([f"{0.1 * i:.1f}" for i in range(1, 10)], rotation=45)

def plot_aggregate_psychometric_right_choice_by_iti(df, ax):
    """
    Plot aggregated psychometric curves over the last 15 days, split by previous ITI duration.
    Each curve shows the average fraction of right responses at each probability level,
    separately for different ITI categories.
    
    Args:
        df (pd.DataFrame): The filtered dataframe.
        ax (matplotlib.axes.Axes): The matplotlib axis where the plot will be drawn.
    """
    # Prepara le date e filtra gli ultimi 15 giorni
    df['date'] = pd.to_datetime(df['date']).dt.date
    last_15_dates = df['date'].dropna().unique()[-15:]
    df_agg = df[df['date'].isin(last_15_dates)].copy()

    if df_agg.empty:
        ax.text(0.5, 0.5, 'No data\n(last 15 days)', ha='center', va='center', fontsize=15)
        ax.set_axis_off()
        return

    # Prepara colonne e variabili
    df_agg['probability_r'] = df_agg['probability_r'].astype(float)
    df_agg['right_choice'] = np.where(df_agg['first_trial_response'] == 'right', 1, 0)
    df_agg['prob_rwd_correct_resp'] = np.where(df_agg['probability_r'] >= 0.5, df_agg['probability_r'], 1 - df_agg['probability_r'])

    df_agg['prev_iti_duration'] = df_agg['iti_duration'].shift(1)
    df_agg['prev_iti_duration'].fillna(0, inplace=True)
    df_agg['prev_iti_category'] = pd.cut(
        df_agg['prev_iti_duration'],
        bins=[-0.1, 2, 6, 12, 30],
        labels=["1-2 sec", "2-6 sec", "7-12 sec", '12-30 sec'],
        ordered=True
    )

    # Aggrega i dati: media per probability_r per ogni categoria di ITI
    grouped = df_agg.groupby(['prev_iti_category', 'probability_r'], observed=True).agg(
        mean_right_choice=('right_choice', 'mean'),
        count=('right_choice', 'count'),
        std=('right_choice', 'std')
    ).reset_index()
    grouped['sem'] = grouped['std'] / grouped['count']**0.5

    iti_categories = grouped['prev_iti_category'].dropna().unique()
    colors = sns.color_palette("viridis", len(iti_categories))
    color_map = dict(zip(iti_categories, colors))

    for category in iti_categories:
        sub = grouped[grouped['prev_iti_category'] == category]
        if sub.empty:
            continue
        # Fit solo sulla media aggregata
        try:
            pars, _ = curve_fit(probit, sub['probability_r'], sub['mean_right_choice'], p0=[1, 0])
            x_fit = np.linspace(0, 1, 100)
            ax.plot(x_fit, probit(x_fit, *pars), '-', color=color_map[category], label=category)
        except RuntimeError:
            print(f"[plot_aggregate_psychometric_right_choice_by_iti] Fit failed for category {category}")

        # Scatter con error bars
        ax.errorbar(sub['probability_r'], sub['mean_right_choice'], yerr=sub['sem'],
                    fmt='o', color=color_map[category], capsize=4)

    ax.axhline(0.5, color='gray', linestyle='--')
    ax.axvline(0.5, color='gray', linestyle='--')
    ax.set_ylim(0, 1)
    ax.set_title('Last 15 days', fontsize=12, pad=10)
    ax.set_xlabel('P(rightward reward)', fontsize=15)
    ax.set_ylabel('Right Choice Rate', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_xticks(np.arange(0.1, 1.0, 0.1))
    ax.set_xticklabels([f"{tick:.1f}" for tick in np.arange(0.1, 1.0, 0.1)], fontsize=13, rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title='ITI duration', fontsize=10, title_fontsize=11, loc='best')

def plot_aggregate_correct_choice_by_iti(df, ax):
    """
    Plot aggregated correct choice curves over the last 15 days, split by previous ITI duration.
    Each curve shows the average fraction of correct responses at each probability level,
    separately for different ITI categories.
    
    Args:
        df (pd.DataFrame): The filtered dataframe.
        ax (matplotlib.axes.Axes): The matplotlib axis where the plot will be drawn.
    """
    # Prepara le date e filtra gli ultimi 15 giorni
    df['date'] = pd.to_datetime(df['date']).dt.date
    last_15_dates = df['date'].dropna().unique()[-15:]
    df_agg = df[df['date'].isin(last_15_dates)].copy()

    if df_agg.empty:
        ax.text(0.5, 0.5, 'No data\n(last 15 days)', ha='center', va='center', fontsize=15)
        ax.set_axis_off()
        return

    df_agg['probability_r'] = df_agg['probability_r'].astype(float)
    df_agg['probability_l'] = 1 - df_agg['probability_r']
    df_agg['prob_rwd_correct_resp'] = np.where(df_agg['probability_r'] >= 0.5, df_agg['probability_r'], 1 - df_agg['probability_r'])

    df_agg['high_choice'] = (
        ((df_agg['first_trial_response'] == 'right') & (df_agg['probability_r'] > df_agg['probability_l'])) |
        ((df_agg['first_trial_response'] == 'left') & (df_agg['probability_l'] > df_agg['probability_r']))
    ).astype(int)

    df_agg['prev_iti_duration'] = df_agg['iti_duration'].shift(1)
    df_agg['prev_iti_duration'].fillna(0, inplace=True)
    df_agg['prev_iti_category'] = pd.cut(
        df_agg['prev_iti_duration'],
        bins=[-0.1, 2, 6, 12, 30],
        labels=["1-2 sec", "2-6 sec", "7-12 sec", '12-30 sec'],
        ordered=True
    )

    grouped = df_agg.groupby(['prev_iti_category', 'prob_rwd_correct_resp'], observed=True).agg(
        mean_correct_choice=('high_choice', 'mean'),
        count=('high_choice', 'count'),
        std=('high_choice', 'std')
    ).reset_index()
    grouped['sem'] = grouped['std'] / grouped['count']**0.5

    iti_categories = grouped['prev_iti_category'].dropna().unique()
    colors = sns.color_palette("viridis", len(iti_categories))
    color_map = dict(zip(iti_categories, colors))

    for category in iti_categories:
        sub = grouped[grouped['prev_iti_category'] == category]
        if sub.empty:
            continue
        try:
            pars, _ = curve_fit(probit, sub['prob_rwd_correct_resp'], sub['mean_correct_choice'], p0=[1, 0])
            x_fit = np.linspace(0, 1, 100)
            ax.plot(x_fit, probit(x_fit, *pars), '-', color=color_map[category], label=category)
        except RuntimeError:
            print(f"[plot_aggregate_correct_choice_by_iti] Fit failed for category {category}")

        ax.errorbar(sub['prob_rwd_correct_resp'], sub['mean_correct_choice'], yerr=sub['sem'],
                    fmt='o', color=color_map[category], capsize=4)

    ax.axhline(0.5, color='gray', linestyle='--')
    ax.axvline(0.5, color='gray', linestyle='--')
    ax.set_ylim(0, 1)
    ax.set_title('Last 15 days', fontsize=12, pad=10)
    ax.set_xlabel('P(rightward reward)', fontsize=15)
    ax.set_ylabel('Fraction of correct responses', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_xticks(np.arange(0.1, 1.0, 0.1))
    ax.set_xticklabels([f"{tick:.1f}" for tick in np.arange(0.1, 1.0, 0.1)], fontsize=13, rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title='ITI duration', fontsize=10, title_fontsize=11, loc='best')

def plot_behavior_around_block_change(df, ax):
    """
    Plot average probability of correct choices around block changes, showing
    behavior adjustment before and after a block switch.

    Args:
        df (pd.DataFrame): The filtered dataframe.
        ax (matplotlib.axes.Axes): The matplotlib axis where the plot will be drawn.
    """

    # Identifica i blocchi
    df['block'] = (df['probability_r'].diff().abs() > 0).cumsum()

    # Calcola la probabilità più alta per ogni trial (per sapere cosa è "corretto")
    df['highest_probability'] = np.where(df['probability_r'] > 0.5, df['probability_r'], 1 - df['probability_r'])

    # Definisci la scelta corretta in base alla porta col reward maggiore
    df['high_choice'] = (
        ((df['first_trial_response'] == 'right') & (df['probability_r'] > 0.5)) |
        ((df['first_trial_response'] == 'left') & (df['probability_r'] < 0.5))
    ).astype(int)

    block_positions, high_choices = [], []

    # Cicla su tutti i blocchi (eccetto il primo)
    for block in df['block'].unique()[1:]:
        block_df = df[df['block'] == block].copy()
        last_trial_prev_block = df[df['block'] == block - 1]['trial'].max()

        block_df['block_position'] = block_df['trial'] - last_trial_prev_block
        block_positions.extend(block_df['block_position'])
        high_choices.extend(block_df['high_choice'])

        # Aggiungi anche gli ultimi 10 trial del blocco precedente
        prev_block_df = df[df['block'] == block - 1].copy()
        prev_block_df = prev_block_df.iloc[-10:] if len(prev_block_df) > 10 else prev_block_df
        prev_block_positions = prev_block_df['trial'] - last_trial_prev_block
        block_positions.extend(prev_block_positions)
        high_choices.extend(prev_block_df['high_choice'])

    # Crea DataFrame con posizione e risposta
    df_block = pd.DataFrame({'block_position': block_positions, 'high_choice': high_choices})

    # Filtro ±10 trial dal cambio
    df_filtered = df_block[df_block['block_position'].between(-10, 9)]

    # Media e SEM
    mean_high = df_filtered.groupby('block_position')['high_choice'].mean()
    sem_high = df_filtered.groupby('block_position')['high_choice'].sem()

    # Plot media + area SEM
    ax.plot(mean_high.index, mean_high, label='Average', color='darkslateblue')
    ax.fill_between(mean_high.index, mean_high - sem_high, mean_high + sem_high, alpha=0.2, color='darkslateblue')

    ax.axvline(x=0, color='black', linestyle='--', label='Block change')
    ax.axhline(y=0.5, color='gray', linestyle='--')

    ax.set_xlabel('Trial Position relative to block change', fontsize=15)
    ax.set_ylabel('P(correct choice)', fontsize=15)
    ax.set_title('Behavior around block change', fontsize=17, pad=10)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, loc='best')

def plot_latency_curves(df, ax_side, ax_center):
    """
    Plot latencies (response times) over days for sides and center pokes.
    Median latencies are plotted in log scale, with task as style.
    
    Args:
        df (pd.DataFrame): The dataframe containing response times and tasks.
        ax_side (matplotlib.axes.Axes): Axis for plotting side latencies.
        ax_center (matplotlib.axes.Axes): Axis for plotting center latencies.
    """
    filtered_df = df[df['reaction_time'].notnull() & df['motor_time'].notnull()].copy()
    
    if filtered_df.empty:
        ax_side.text(0.5, 0.5, 'No latency data', ha='center', va='center', fontsize=15)
        ax_side.set_axis_off()
        ax_center.set_axis_off()
        return

    # Plot side latencies
    sns.lineplot(
        x='date', y='motor_time', style='task', data=filtered_df,
        estimator=np.median, errorbar=None, ax=ax_side, color='black', markers=True
    )
    ax_side.set_yscale('log')
    ax_side.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    for y in [0, 1, 10]:
        ax_side.axhline(y=y, linestyle='--', color='lightgray', alpha=0.7)
    ax_side.tick_params(axis='x', rotation=45, labelsize=14)
    ax_side.set_ylabel('Sides-Latency (MT)', fontsize=14)
    ax_side.set_xlabel('')
    ax_side.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax_side.spines['top'].set_visible(False)
    ax_side.spines['right'].set_visible(False)
    ax_side.get_legend().remove()

    # Plot center latencies
    sns.lineplot(
        x='date', y='reaction_time', style='task', data=filtered_df,
        estimator=np.median, errorbar=None, ax=ax_center, color='black', markers=True
    )
    ax_center.set_yscale('log')
    ax_center.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    for y in [0, 1, 10]:
        ax_center.axhline(y=y, linestyle='--', color='lightgray', alpha=0.7)
    ax_center.tick_params(axis='x', rotation=45,  labelsize=14)
    ax_center.set_xlabel('Day', fontsize=14)
    ax_center.set_ylabel('Center-Latency (RT)', fontsize=14)
    ax_center.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax_center.spines['top'].set_visible(False)
    ax_center.spines['right'].set_visible(False)
    ax_center.get_legend().remove()

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
    if 'iti_duration' not in df.columns or df['iti_duration'].dropna().empty:
        ax.text(0.5, 0.5, 'No ITI data', ha='center', va='center', fontsize=12)
        ax.set_axis_off()
        return

    iti_data = df['iti_duration'].dropna()

    ax.hist(iti_data, bins=bins, color='lightseagreen', edgecolor='black', alpha=0.8)
    ax.set_title("ITI Duration Histogram", fontsize=16, pad=10)
    ax.set_xlabel("ITI Duration (s)", fontsize=14)
    ax.set_ylabel("Number of Trials", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_latency_across_trials(df, ax_side, ax_center):
    """
    Plot latencies across trials (not days), to visualize engagement within sessions.
    Plots latencies for side and center responses aligned by trial number.
    
    Args:
        df (pd.DataFrame): Dataframe with response times and trial numbers.
        ax_side (matplotlib.axes.Axes): Axis for side latency plot.
        ax_center (matplotlib.axes.Axes): Axis for center latency plot.
    """
    # Filter valid latencies
    filtered_df = df[df['reaction_time'].notnull() & df['motor_time'].notnull()].copy()

    if filtered_df.empty:
        ax_side.text(0.5, 0.5, 'No latency data', ha='center', va='center', fontsize=15)
        ax_side.set_axis_off()
        ax_center.set_axis_off()
        return

    # Side latency vs trial
    sns.lineplot(
        x='trial', y='motor_time', hue='task', data=filtered_df,
        estimator='median', errorbar=None, ax=ax_side, markers=True
    )
    ax_side.set_yscale('log')
    ax_side.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    for y in [0, 1, 10]:
        ax_side.axhline(y=y, linestyle='--', color='lightgray', alpha=0.7)
    ax_side.set_xlabel('Trial', fontsize=14)
    ax_side.set_ylabel('Side-Latency (RT)', fontsize=14)
    ax_side.tick_params(axis='both', labelsize=14)
    ax_side.spines['top'].set_visible(False)
    ax_side.spines['right'].set_visible(False)
    ax_side.legend(fontsize=8, loc='upper right', title='Task')

    # Center latency vs trial
    sns.lineplot(
        x='trial', y='reaction_time', hue='task', data=filtered_df,
        estimator='median', errorbar=None, ax=ax_center, markers=True
    )
    ax_center.set_yscale('log')
    ax_center.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    for y in [0, 1, 10]:
        ax_center.axhline(y=y, linestyle='--', color='lightgray', alpha=0.7)
    ax_center.set_xlabel('Trial', fontsize=14)
    ax_center.set_ylabel('Center-Latency (RT)', fontsize=14)
    ax_center.tick_params(axis='both', labelsize=14)
    ax_center.spines['top'].set_visible(False)
    ax_center.spines['right'].set_visible(False)
    ax_center.legend(fontsize=10, loc='upper right', title='Task')

def plot_sessionwise_binned_outcome_aggregate(df, ax, bin_size=10, min_trials_per_bin=5, smooth_window=3):
    """
    Plot the average outcome (fraction of correct responses) binned across all sessions,
    aligned by trial number, with optional smoothing.
    
    Args:
        df (pd.DataFrame): DataFrame con colonne 'trial' e 'fraction_of_correct_responses'.
        ax (matplotlib.axes.Axes): Axis su cui plottare.
        bin_size (int): Numero di trial per bin.
        min_trials_per_bin (int): Minimo numero di trial richiesti per mantenere un bin.
        smooth_window (int): Dimensione finestra per smoothing con media mobile.
    """
    df = df.copy()
    df['trial_bin'] = (df['trial'] // bin_size) * bin_size

    # Conta i trial per bin
    bin_counts = df.groupby('trial_bin').size().reset_index(name='n_trials')

    # Calcola la media dell'outcome per bin
    outcome_mean = df.groupby('trial_bin')['prob_rwd_correct_resp'].mean().reset_index()

    # Merge per aggiungere la numerosità
    merged = pd.merge(outcome_mean, bin_counts, on='trial_bin')

    # Filtra via i bin con pochi trial
    merged = merged[merged['n_trials'] >= min_trials_per_bin]

    if merged.empty:
        ax.text(0.5, 0.5, 'No data for outcome plot', ha='center', va='center', fontsize=15)
        ax.set_axis_off()
        return

    # Smoothing con media mobile
    smoothed_outcome = merged['prob_rwd_correct_resp'].rolling(smooth_window, center=True).mean()

    # Plot
    ax.plot(merged['trial_bin'], smoothed_outcome, label='Avg Outcome', color='black', marker='o')
    ax.bar(merged['trial_bin'], merged['n_trials'] / merged['n_trials'].max(),
           width=bin_size * 0.9, color='lightgray', alpha=0.4, label='Trial count')

    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Trial in session', fontsize=15)
    ax.set_ylabel('Fraction of correct responses', fontsize=15)
    ax.set_title('Session-wise Avg Outcome', fontsize=15)
    ax.legend(fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
