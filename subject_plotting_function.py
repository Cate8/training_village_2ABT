import pandas as pd
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from plotting_functions import *
from session_parsing_functions import *
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

def plot_number_of_sessions(df, ax, label_kwargs=None):
    """
    Plots the number of sessions per day with different styles per task.

    Parameters:
    - df: DataFrame with at least columns 'day', 'session', 'task'
    - ax: Matplotlib axis to draw the plot on
    - label_kwargs: Optional dict for customizing axis labels
    """
    label_kwargs = label_kwargs or {}

    # Group sessions per day and task
    sessions_df = df.groupby(['date']).agg({
        'session': 'nunique',
        'task': 'max'
    }).reset_index()

    # Plot with styles per task
    sns.lineplot(
        data=sessions_df,
        x='date', y='session',
        hue='task', style='task',
        markers=True, dashes=True,
        ax=ax, palette='tab10'
    )

    # Reference line
    ax.axhline(y=3, color='gray', linestyle=':', linewidth=1)

    # Format axes
    ax.set_ylabel('Nº of sessions', **label_kwargs)
    ax.set_xlabel('Day', **label_kwargs)
    ax.tick_params(axis='x', rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # Legend
    ax.legend(title='Task', fontsize=8)

    return ax

    
def plot_trials_and_water(df, ax, label_kwargs=None):
    """
    Plots the number of trials and the amount of water drunk per day over the last 15 days.

    Parameters:
    - df: pandas DataFrame with at least the following columns:
        'date', 'session', 'trial', 'task', and 'water'
    - ax: Matplotlib axis where the plot will be rendered
    - label_kwargs: Optional dict of label customization
    """

    label_kwargs = label_kwargs or {}

    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Group by session to get daily trial counts
    trial_df = df.groupby('session').agg({
        'trial': 'max',
        'date': 'max',
        'task': 'max'
    }).reset_index()

    # Group by session to get total water per session
    daily_df = df.groupby(['date', 'session']).agg({
        'water': 'sum',
        'task': 'max'
    }).reset_index()

    # Aggregate trial counts and water by day
    grouped_trials = trial_df.groupby('date').agg({'trial': 'sum'}).reset_index()
    grouped_water = daily_df.groupby('date').agg({'water': 'sum'}).reset_index()

    # Merge trials and water on 'date'
    grouped_df = pd.merge(grouped_trials, grouped_water, on='date', how='outer').fillna(0)

    # Keep only the last 15 days
    grouped_df = grouped_df[grouped_df['date'] >= grouped_df['date'].max() - pd.Timedelta(days=15)]

    # Plot trial data
    sns.lineplot(
        x='date', y='trial',
        data=grouped_df,
        ax=ax, color='black', marker='o', label='Trials/day'
    )

    # Add second axis for water
    ax2 = ax.twinx()
    sns.lineplot(
        x='date', y='water',
        data=grouped_df,
        ax=ax2, color='blue', marker='s', label='Water (ul)/day'
    )

    # Axis formatting
    ax.set_ylabel('Nº of trials (µ)', **label_kwargs)
    ax.set_ylim(0, 350)
    ax.set_xlabel('')
    ax.xaxis.set_ticklabels([])

    ax2.set_ylabel('Water (ul)')

    # Reference line
    ax.axhline(y=170, color='gray', linestyle=':', linewidth=1)

    # Water label text
    try:
        recent = grouped_df.sort_values('date').iloc[-3:]
        label = ', '.join([
            f"{date.strftime('%b %d')}: {int(w)}ul"
            for date, w in zip(recent['date'], recent['water'])
        ])
    except:
        label = "Water info not available"

    ax.text(0.12, 1.2, f"Water Last Days → {label}", transform=ax.transAxes,
            fontsize=8, fontweight='bold', verticalalignment='top')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax

def clean_time_column(col):
    def convert(val):
        if pd.isna(val) or val == "":
            return np.nan
        try:
            parsed = ast.literal_eval(val) if isinstance(val, str) and val.startswith("[") else val
            if isinstance(parsed, list) and parsed:
                return float(parsed[0])
            return float(parsed)
        except Exception:
            return np.nan
    return col.apply(convert)

def get_task_stage(df):
    # Inferisci il task in base alle colonne presenti e non NaN
    if 'STATE_c_led_on_START' in df.columns and df['STATE_c_led_on_START'].notna().any():
        return "S3_or_S4"
    elif 'TRIAL_START' in df.columns and df['TRIAL_START'].notna().any():
        return "S1_or_S2"
    else:
        return "other"

def plot_cumulative_trial_rate(df, axes, label_kwargs=None, palette_name='Greens', n_days=5):
    stage = get_task_stage(df)

    if stage == "S3_or_S4":
        start_col = 'STATE_c_led_on_START'
        end_col = 'STATE_drink_delay_END'
    elif stage == "S1_or_S2":
        start_col = 'TRIAL_START'
        end_col = 'TRIAL_END'
    else:
        print(f"[Warning] Task stage '{stage}' non gestito. Skipping plot.")
        return

    if start_col not in df.columns or end_col not in df.columns:
        print(f"[Warning] Colonne {start_col} o {end_col} mancanti nel DataFrame.")
        return

    # Pulisci i tempi
    df[start_col] = clean_time_column(df[start_col])
    df[end_col] = clean_time_column(df[end_col])

    df = df.dropna(subset=[start_col, end_col])
    if df.empty:
        print("[Warning] Tutte le righe con tempi validi sono NaN. Nessun dato utile per il plot.")
        return

    # Calcolo durate e tempo corrente
    df['start_session'] = df.groupby(['subject', 'session'])[start_col].transform('min')
    df['end_session'] = df.groupby(['subject', 'session'])[end_col].transform('max')
    df['session_lenght'] = (df['end_session'] - df['start_session']) / 60
    df['current_time'] = df.groupby(['subject', 'session'])[start_col].transform(lambda x: (x - x.iloc[0]) / 60)

    if df['session_lenght'].dropna().empty:
        print("[Warning] Tutte le sessioni hanno durata NaN. Nessun dato utile per il plot.")
        return

    max_timing = int(round(df['session_lenght'].max()))
    sess_palette = sns.color_palette(palette_name, n_days)

    if 'day' not in df.columns:
        df['day'] = df['session']  # fallback

    last_days = df['day'].unique()[-n_days:]

    for idx, day in enumerate(last_days):
        subset = df[df['day'] == day]
        n_sess = subset['session'].nunique()

        if n_sess == 0 or subset['current_time'].dropna().empty:
            print(f"[Warning] Giorno {day} senza dati validi.")
            continue

        try:
            hist_ = stats.cumfreq(
                subset['current_time'],
                numbins=max_timing,
                defaultreallimits=(0, subset['current_time'].max())
            )
        except Exception as e:
            print(f"Fallback per giorno {day} a causa di: {e}")
            hist_ = stats.cumfreq(
                subset['current_time'],
                numbins=max_timing,
                defaultreallimits=(0, max_timing)
            )

        hist_norm = hist_.cumcount / n_sess
        bins_plt = hist_.lowerlimit + np.linspace(0, hist_.binsize * hist_.cumcount.size, hist_.cumcount.size)

        sns.lineplot(
            x=bins_plt, y=hist_norm,
            color=sess_palette[idx % len(sess_palette)],
            ax=axes, marker='o', markersize=4
        )

    axes.set_ylabel('Cum. nº of trials', label_kwargs or {})
    axes.set_xlabel('Time (mins)', label_kwargs or {})

    legend_labels = np.arange(-n_days, 0, 1)
    lines = [
        Line2D([0], [0], color=sess_palette[i], marker='o', markersize=7, markerfacecolor=sess_palette[i])
        for i in range(len(legend_labels))
    ]
    axes.legend(lines, legend_labels, title='Days', loc='center', bbox_to_anchor=(0.1, 0.85))
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)

def plot_calendar(ax, df):
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

def plot_task_histogram(ax, df):
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

def plot_session_count_by_task(ax, df):
    """
    Plot histogram showing the number of unique sessions per task type.
    Uses consistent colors with the calendar plot.
    """
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

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_activity_timeline(ax, df):
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
