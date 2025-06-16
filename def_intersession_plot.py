import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
from utils_functions import *
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
import seaborn as sns


def get_task_stage(task_str: str) -> str:
    """
    Classify task name into stage categories (S3, S4, other).
    """
    if task_str.startswith("S3"):
        return "S3"
    elif task_str.startswith("S4"):
        return "S4"
    else:
        return "other"


def parse_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataframe: parse dates, assign ports, define task_stage.
    """
    df = df.copy()

    # Ensure 'date' column is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    # Assign poke ports based on system
    if "system_name" in df.columns:
        df = assign_ports(df)

    # Add task_stage
    if "task" in df.columns:
        df["task_stage"] = df["task"].apply(get_task_stage)

    def safe_extract_first_float(column):
        try:
            return extract_first_float(column)  # Tentativo di estrarre il valore
        except Exception as e:
            print(f"Warning: Error processing column {column.name} -> {e}")  # Log dell'errore
            return None  # Restituisce None se si verifica un errore

    # Applicazione della funzione sicura per tutte le colonne
    df['first_response_right'] = df['right_poke_in'].apply(safe_extract_first_float)
    df['first_response_left'] = df['left_poke_in'].apply(safe_extract_first_float)

    if 'centre_poke_in' in df.columns:
        df['first_response_center'] = df['centre_poke_in'].apply(safe_extract_first_float)
    else:
        df['first_response_center'] = None 

    conditions = [
        df['first_response_left'].isna() & df['first_response_right'].isna(),
        df['first_response_left'].isna(),
        df['first_response_right'].isna(),
        df['first_response_left'] <= df['first_response_right'],
        df['first_response_left'] > df['first_response_right'],
    ]
    choices = ["no_response", "right", "left", "left", "right"]
    df["first_trial_response"] = np.select(conditions, choices)


    return df


def filter_last_days(df: pd.DataFrame, days: int = 50) -> pd.DataFrame:
    """
    Filter DataFrame to keep only rows from the last N days.
    """
    if "date" not in df.columns:
        print("[INFO] No 'date' column – using all data.")
        return df.copy()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        print("[INFO] No valid dates – using all data.")
        return df

    cutoff = datetime.now().date() - timedelta(days=days)
    recent_df = df[df["date"].dt.date >= cutoff]

    if recent_df.empty:
        print("[INFO] No data in last days – using all data.")
        return df

    return recent_df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
import seaborn as sns

def generate_psychometric_plot(df: pd.DataFrame, output_pdf_path: str = None) -> None:
    """
    Generate psychometric plots: Right Choice Reward vs Probability Type and
    Right Choice Reward vs Probability Type splitted by ITIs, but only for the task 'S4'.
    
    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing columns: 'subject', 'task', 'probability_r', 
        'first_trial_response', 'day', 'iti_duration'.
    output_pdf_path : str, optional
        Path to save the plot as a PDF file.
    """
    
    # Funzione probit per adattamento della curva
    def probit(x, beta, alpha):
        return 0.5 * (1 + erf((beta * x + alpha) / np.sqrt(2)))

    # Verifica che il dataset contenga le colonne richieste
    if not all(col in df.columns for col in ['task', 'probability_r', 'first_trial_response', 'day', 'iti_duration']):
        raise ValueError("Il DataFrame non contiene tutte le colonne richieste!")

    # Filtro per includere solo i dati del task 'S4'
    df_s4 = df[df['task'] == 'S4_0']
    
    # Se non ci sono dati per il task 'S4', termina la funzione
    if df_s4.empty:
        print("[WARNING] No data for task 'S4'. Skipping the plotting.")
        return

    # Filtro per escludere il 'manual'
    subjects = df_s4['subject'].unique()
    subjects = subjects[subjects != 'manual']

    # --- PSICOMETRICA: RIGHT CHOICE REWARD VS PROBABILITY TYPE ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # Plot con 2 colonne

    # Prima sottotrama: right choice reward vs probability type
    ax = axes[0]

    for subject in subjects:
        df_subject = df_s4[df_s4['subject'] == subject]
        df_subject['probability_l'] = 1 - df_subject['probability_r']
        df_subject['probability_r'] = df_subject['probability_r'].round(1)
        df_subject['probability_l'] = df_subject['probability_l'].round(1)

        probs = np.sort(df_subject['probability_r'].unique())
        right_choice_freq = []

        for prob in probs:
            indx_blk = df_subject['probability_r'] == prob
            sum_prob = np.sum(indx_blk)
            indx_blk_r = indx_blk & (df_subject['first_trial_response'] == 'right')
            sum_choice_prob = np.sum(indx_blk_r)
            if sum_prob > 0:  # Solo se ci sono dati per quella probabilità
                right_choice_freq.append(sum_choice_prob / sum_prob)

        # Fit probit
        pars, _ = curve_fit(probit, df_subject['probability_r'], df_subject['first_trial_response'] == 'right', p0=[0, 1])

        # Plot curva psicometrica
        x = np.linspace(0, 1, 100)
        ax.plot(x, probit(x, *pars), label=f'{subject}')
        ax.scatter(probs, right_choice_freq, marker='o')

    # Dettagli del grafico
    ax.axhline(0.5, color='gray', linestyle='--')
    ax.axvline(0.5, color='gray', linestyle='--')
    ax.set_xlabel('Probability Type')
    ax.set_ylabel('Right Choice Rate')
    ax.set_title('Right Choice Reward vs Probability Type')
    ax.set_ylim(0, 1)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    sns.despine()

    # --- PSICOMETRICA: RIGHT CHOICE REWARD VS PROBABILITY TYPE SPLITTED BY ITIs ---
    ax = axes[1]

    # Prepara i dati per analizzare la frazione di risposte giuste in base a ITI
    df3 = df_s4[['probability_r', 'first_trial_response', 'iti_duration']].copy()
    df3['right_choice'] = np.where(df3['first_trial_response'] == 'right', 1, 0)
    df3['prev_iti_duration'] = df3['iti_duration'].shift(1)
    df3['prev_iti_duration'].fillna(0, inplace=True)  # Sostituisce NaN con 0
    df3['prev_iti_category'] = pd.cut(df3['prev_iti_duration'], 4,
                                      labels=["1-2 sec", "2-6 sec", "7-12 sec", '12-30 sec'])

    # Aggrega i dati
    grouped3 = df3.groupby(['prev_iti_category', 'probability_r']).agg(
        fraction_of_right_responses=('right_choice', 'mean')
    ).reset_index()

    # Colormap per le categorie di ITI
    iti_categories = grouped3['prev_iti_category'].unique()
    num_colors = len(iti_categories)
    colors = sns.color_palette("viridis", num_colors)
    color_map = dict(zip(iti_categories, colors))

    for category in iti_categories:
        subset = grouped3[grouped3['prev_iti_category'] == category]
        x_data = subset['probability_r']
        y_data = subset['fraction_of_right_responses']

        if len(x_data) > 0 and len(y_data) > 0:  # Assicurati che ci siano abbastanza dati
            # Fit probit
            pars, _ = curve_fit(probit, x_data, y_data, p0=[1, 0])
            # Plot la curva adattata
            x_fit = np.linspace(0, 1, 100)
            y_fit = probit(x_fit, *pars)
            ax.plot(x_fit, y_fit, '-', color=color_map[category], label=f'{category}')
            ax.scatter(x_data, y_data, color=color_map[category])  # Scatter dei dati reali

    # Dettagli del grafico
    ax.axhline(0.5, color='gray', linestyle='--')
    ax.axvline(0.5, color='gray', linestyle='--')
    ax.set_xlabel('Probability Reward Right Response')
    ax.set_ylabel('Fraction Right Responses')
    ax.set_title('Right Choice Reward vs Probability Type (by ITI Duration)')
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    ax.legend(title='ITI Duration')
    ax.set_ylim(0, 1)
    sns.despine()

    # Mostra il plot
    plt.tight_layout()

    # Salva il grafico in PDF (opzionale)
    if output_pdf_path:
        with plt.backends.backend_pdf.PdfPages(output_pdf_path) as pdf:
            pdf.savefig(fig)  # Salva il plot come pagina nel PDF
            plt.close(fig)  # Chiude il plot per evitare conflitti
    else:
        plt.show()  # Mostra il grafico
