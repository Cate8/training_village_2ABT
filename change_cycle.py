# runs when cycle day/night changes and there is no animal in the behavioural box
"""
1. Deletes the old videos.
2. You can override this class in your project code to implement custom behavior.
"""

from pathlib import Path

from village.scripts.safe_removal_of_data import main as safe_removal_script
from village.settings import Active, settings
from village.classes.change_cycle_run import ChangeCycleRun
from send_intersession_slack import send_slack_plots
import time


class ChangeCycle(ChangeCycleRun):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        # si es antes de las 12 de la mañana, envía los plots al slack
        now = time.time()
        #if time.localtime(now).tm_hour < 12:
        if True:
            send_slack_plots()

        safe_removal_script(
            directory=self.directory,
            days=self.days,
            safe=self.safe,
            backup_dir=self.backup_dir,
            remote_user=self.remote_user,
            remote_host=self.remote_host,
            port=self.port,
        )

