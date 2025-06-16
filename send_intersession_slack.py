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
from def_intersession_plot import *
from utils_functions import *

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

# -- plotting --------------------------------------------------------------

def plot_dataframe(df: pd.DataFrame, title: str, output_pdf_path) -> None:
    """
    Main function: parses, filters, computes and plots into A4 PDF.
    """
    df = parse_dataframe(df)
    df = filter_last_days(df)

    #print(f"Columns of the DataFrame used for plotting: {df.columns.tolist()}")
    with PdfPages(output_pdf_path) as pdf:
        fig, axs = plt.subplots(2, 1, figsize=(8.27, 11.69))  # A4 portrait in inches

        # Plot accuracy by stage
        generate_psychometric_plot(df)

        # Optional second plot (empty grid placeholder)
        axs[1].axis("off")
        axs[1].text(0.5, 0.5, "Second plot goes here", ha="center", va="center")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

def send_slack_plots() -> None:
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
        plot_dataframe(df, f"{animal_name} ({DATE_STR})", output_pdf)

        slack_spam(
            msg=f"Intersession for {animal_name}",
            filepath=str(output_pdf),
            userid=SLACK_CHANNEL,
        )

if __name__ == "__main__":
    send_slack_plots()
