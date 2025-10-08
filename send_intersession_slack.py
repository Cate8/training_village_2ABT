"""
Automatically generates plots from the **aggregated CSV per animal** and
uploads them to Slack via a helper `slack_spam()`.

Key points
----------
* Accepts **channel/user IDs** that start with `C` or `U` _directly_.
* Allows mapping from friendly names (e.g. `#prl_reports` or `jordi`) to IDs
  via an internal dictionary.
* Uses the modern `files_upload_v2()` endpoint to avoid deprecation issues.
* Loads credentials from a `.env` file with `python-dotenv`.

Required environment variables
------------------------------
* `SLACK_BOT_TOKEN` – bot token (`xoxb-…`) with `files:write`/`chat:write`.
* `SLACK_CHANNEL`   – **ID** of the default channel (e.g. `C0123456789`).
"""
from __future__ import annotations
import os
import re
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import slack_sdk as slack
from dotenv import load_dotenv
from Plotting_intersession import *
from session_parsing_functions import *
from utils_functions import *
from plotting_functions import *
from matplotlib import gridspec


# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
BASE_DIR = Path("/home/pi/village_projects/cate_task/data/sessions")
DATE_STR = datetime.now().strftime("%Y%m%d")  # e.g. 20250611
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "C0123456789")  # fallback ID
SLACK_TOKEN = os.getenv("SLACK_BOT_TOKEN")
# ---------------------------------------------------------------------------

def slack_spam(msg: str, filepath: str | None, userid: str) -> None:
    """Send a message or file to *userid* via the bot.

    Parameters
    ----------
    msg : str
        Text message or initial comment for the file.
    filepath : str | None
        Path to the file to upload. If *None*, only a text message is sent.
    userid : str
        Slack recipient – can be a channel/user ID (`C…`, `U…`), a channel
        name starting with `#`, or a nickname present in *ids_dic*.
    """
    ids_dic = {
        "#prl_reports": "C06UZ1WFMAT", # PRL reports channel
        "jordi": "U8J8YA66S",
        "lejla": "U7TTEEN4T",
        "dani": "UCFMZDWE8",
        "yerko": "UB3B8425D",
        "carles": "UPZPM32UC",
    }

    # Resolve *userid* to a valid Slack ID -------------------------------
    if userid.startswith(("U", "C")):
        slack_id = userid  # already a valid ID
    elif userid.startswith("#"):
        slack_id = ids_dic.get(userid)
        if slack_id is None:
            raise ValueError(f"Unknown Slack channel name: {userid}")
    else:
        slack_id = ids_dic.get(userid.lower())
        if slack_id is None:
            raise ValueError(f"Unknown Slack nickname: {userid}")

    # Token check --------------------------------------------------------
    if not SLACK_TOKEN:
        raise EnvironmentError("SLACK_BOT_TOKEN is not set")

    client = slack.WebClient(token=SLACK_TOKEN)

    # Send ---------------------------------------------------------------
    if filepath and Path(filepath).exists():
        client.files_upload_v2(
            channel=slack_id,
            file=filepath,
            title=os.path.basename(filepath),
            initial_comment=msg,
        )
    else:
        client.chat_postMessage(channel=slack_id, text=msg)

# -- utilities -------------------------------------------------------------
AGGR_REGEX = re.compile(r"(all|combined|aggregate|tutte|all_sessions)", re.I)

def find_aggregate_csv(animal_dir: Path) -> Path | None:
    """Return the aggregate CSV within *animal_dir*, specifically matching the pattern."""
    csv_paths = list(animal_dir.glob("*.csv"))  # List all CSV files in the directory
    if not csv_paths:
        return None
    
    # Look for the exact file (e.g., 'RATON2.csv')
    for p in csv_paths:
        if p.name.lower() == f"{animal_dir.name.lower()}.csv":  # Check for matching name
            return p

    # If no exact match, return None or you can handle differently
    return None


# -- plotting Last Stages S3 and S4 --------------------------------------------------------------
def plot_dataframe(df: pd.DataFrame, title: str, output_pdf_path: str) -> None:
    """
    Generate a multipanel plot in an A4 PDF using GridSpec with a custom layout,
    only for specific tasks (S3, S4_0, S4_1, ...).
    """
    # Filter dataframe to include only tasks of interest
    tasks_to_plot = ['S3'] + [f'S4_{i}' for i in range(10)]  # Customize the range as needed
    df = df[df['task'].isin(tasks_to_plot)].copy()
    print("[DEBUG] Columns after reading CSV:", list(df.columns))
    print(f"[DEBUG] Unique tasks after filter: {df['task'].unique()}")
    print(f"[DEBUG] Number of rows after filter: {len(df)}")
    if df.empty:
        print("No data available for the selected tasks.")
        return

    # Assign standardized port columns
    df = assign_ports_intersession(df)
    print("[DEBUG] Columns after assign_ports:", list(df.columns))

    # Check required columns after assign_ports
    required_columns = ['centre_poke_in']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"[Main] Required columns missing after assign_ports: {missing_columns}. Skipping parsing and plotting.")
        return

    # Parse data for S4
    df = parse_S4_data(df)

    # Convert date column to datetime for analysis
    df_aggregated = aggregate_data(df)
    print(f"[DEBUG] Sample of df_aggregated['date']:\n{df_aggregated['date'].head()}")
    print(f"[DEBUG] Unique dates in df_aggregated: {df_aggregated['date'].unique()}")

    # Create the PDF and the figure with A4 size
    with PdfPages(output_pdf_path) as pdf:
        fig = plt.figure(figsize=(20, 27))  # A4 portrait in inches

        # GridSpec with 5 rows and 5 columns for flexible layout
        gs = gridspec.GridSpec(
            6, 5,
            figure=fig,
            height_ratios=[0.1, 1.3, 1.3, 1.3, 1.5, 1.5],
            width_ratios=[1.3, 1.3, 1.3, 1.3, 1.3]
        )

        # --- ROW 0: Subject title ---
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis("off")
        ax_title.text(0.5, 0.5, title, fontsize=16, va='center', ha='center', weight='bold')

        # --- ROW 1: 5 psychometric plots ---
        ax_row1 = [fig.add_subplot(gs[1, i]) for i in range(5)]
        print(f"[DEBUG] Calling plot_psychometric_curves with {len(df_aggregated)} rows")
        plot_psychometric_curves(df_aggregated, ax_row1)

        # --- ROW 2: 5 plots ---
        ax_row2 = [fig.add_subplot(gs[2, i]) for i in range(5)]
        plot_correct_choice_curves(df_aggregated, ax_row2)

        # --- ROW 3: 4 plots ---
        ax_row3 = [fig.add_subplot(gs[3, i]) for i in range(5)]
        plot_aggregate_psychometric_right_choice(df, ax_row3[0])
        plot_aggregate_correct_choice_mean(df, ax_row3[1])
        plot_aggregate_psychometric_right_choice_by_iti(df, ax_row3[2])
        plot_aggregate_correct_choice_by_iti(df, ax_row3[3])
        plot_behavior_around_block_change(df, ax_row3[4])

        # --- ROW 4: 3 long plot ---
        ax_rowN_side = fig.add_subplot(gs[4, 0:2])  # es. prima metà della riga
        ax_rowN_center = fig.add_subplot(gs[4, 2:4])  # seconda metà della riga
        plot_latency_curves(df, ax_rowN_side, ax_rowN_center)
        ax_iti = fig.add_subplot(gs[4, 4:5])  # occupa la riga 8, colonne 0-50
        plot_iti_histogram(ax_iti, df)

        # --- ROW 5: 3 long plot ---
        ax_lastrow_side = fig.add_subplot(gs[5, 0:2])
        ax_lastrow_center = fig.add_subplot(gs[5, 2:4])
        plot_latency_across_trials(df, ax_lastrow_side, ax_lastrow_center)

        ax_session_agg = fig.add_subplot(gs[5, 4:5])
        plot_sessionwise_binned_outcome_aggregate(df, ax_session_agg, bin_size=10, min_trials_per_bin=5, smooth_window=3)

        # Layout adjustments
        plt.subplots_adjust(hspace=0.5, wspace=0.4, top=0.95, bottom=0.05, left=0.05, right=0.95)
    
        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF saved to {output_pdf_path}")


def send_slack_plots() -> None:
    print("slack plots")
    for animal_dir in sorted(p for p in BASE_DIR.iterdir() if p.is_dir()):
        csv_path = find_aggregate_csv(animal_dir)
        animal_name = animal_dir.name
        output_pdf = animal_dir / f"{animal_name}_{DATE_STR}.pdf"

        print(f"Reading CSV from: {csv_path}")

        if csv_path is None:
            print(f"[WARNING] {animal_dir}: no aggregate CSV found – blank PDF")
            #generate_blank_pdf(f"{animal_name} ({DATE_STR})", output_pdf)
            slack_spam(
                msg=f"Blank PDF for {animal_name}",
                filepath=str(output_pdf),
                userid=SLACK_CHANNEL,
            )
            continue

        try:
            df = pd.read_csv(csv_path, sep=';')
        except Exception as exc:
            print(f"[ERROR] {csv_path}: {exc} – blank PDF")
            #generate_blank_pdf(f"{animal_name} ({DATE_STR})", output_pdf)
            slack_spam(
                msg=f"(Error reading CSV) {animal_name}",
                filepath=str(output_pdf),
                userid=SLACK_CHANNEL,
            )
            continue

        print(f"[.] {animal_name}: generating plot → {output_pdf.name}")
        try:
            plot_dataframe(df, f"{animal_name} ({DATE_STR})", output_pdf)
        except Exception as e:
            print(e)

        slack_spam(
            msg=f"Intersession for {animal_name}",
            filepath=str(output_pdf),
            userid=SLACK_CHANNEL,
        )

if __name__ == "__main__":
    send_slack_plots()
