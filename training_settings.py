from village.classes.training import Training


class TrainingSettings(Training):
    """
    This class defines how the training protocol is going to be.
    This is, how variables change depending on different conditions (e.g. performance),
    and/or which tasks are going to be run.

    In this class 2 methods need to be implemented:
    - __init__
    - update_training_settings

    In __init__ all the variables that can modify the state of the training protocol
    must be defined.
    When a new subject is created, a new row is added to the data/subjects.csv file,
    with these variables and its values.

    The following variables are needed:
    - self.next_task
    - self.refractary_period
    - self.minimum_duration
    - self.maximum_duration
    In addition to these variables, all the necessary variables to modify the state
    of the tasks can be included.

    When a task is run the values of the variables are read from the json file.
    When the task ends, the values of the variables are updated in the json file,
    following the logic in the update method"""

    def __init__(self) -> None:
        super().__init__()

    def default_training_settings(self) -> None:
        """
        This method is called when a new subject is created.
        It sets the default values for the training protocol.
        """
        # Settings in this block are mandatory for everything
        # that runs on Traning Village
        # TODO: explain them
        self.settings.next_task = "Habituation"
        self.settings.refractary_period = 14400
        self.settings.minimum_duration = 600
        self.settings.maximum_duration = 3600

        # Settings in this block are dependent on each task,
        # and the user needs to create and define them here
        self.settings.middle_port_light_intensity = 50
        self.settings.timer_for_response = 5
        self.settings.iti = 1
        self.settings.reward_amount_ml = 5
        self.settings.punishment = False
        self.settings.punishment_time = 1
        self.settings.trial_types = ["left_easy", "right_easy"]
        self.settings.side_port_light_intensities = [100, 200]

        self.settings.a = 1
        self.settings.b = 2
        self.settings.c = 3
        self.settings.d = 4
        self.settings.e = 5
        self.settings.f = 6
        self.settings.g = 7
        self.settings.h = 8
        self.settings.i = 9
        self.settings.j = 10
        self.settings.k = 11
        self.settings.l = 12
        self.settings.m = 13
        self.settings.n = 14
        self.settings.o = 15
        self.settings.p = 16
        self.settings.q = 17
        self.settings.r = 18
        self.settings.s = 19
        self.settings.t = 20
        self.settings.u = 21
        self.settings.v = 22
        self.settings.w = 23
        self.settings.x = 24
        self.settings.y = 25
        self.settings.z = 26

    def update_training_settings(self) -> None:
        """
        This method is called every time a session finishes.
        It is used to make the animal progress in the training protocol.

        For this example, we want the animal to go from Habituation to FollowTheLight
        after 2 sessions, as long as it completed overall more than 100 trials.
        We also want to decrease the reward amount during the first sessions.
        We promote the animals to the second training stage in FollowTheLight
        when they do two consecutive sessions with over 85% performance.
        Note that in this case, they never go back to the easier task.
        """
        ## You have access to the following variables:
        # self.last_task contains the name of the task that the mouse did last
        # self.subject contains the name of the mouse
        # self.df object contains all data from training for a particular subject
        # self.settings contains the settings from the last session

        # If the last task is Simple, return none
        if self.last_task == "Simple":
            return None

        # get some information
        total_trials = self.df.shape[0]
        total_sessions = len(self.df.session.unique())

        # define when to change tasks
        if self.last_task == "Habituation" and total_trials > 100 and total_sessions >= 2:
            self.settings.next_task = "FollowTheLight"

        # decrease the reward amount at the beginning of training
        match total_sessions:
            case 0:
                self.settings.reward_amount_ml = 5
            case 1:
                self.settings.reward_amount_ml = 4
            case 2:
                self.settings.reward_amount_ml = 3.5
            case 3:
                self.settings.reward_amount_ml = 3
            case 4:
                self.settings.reward_amount_ml = 2.5
            case _:
                self.settings.reward_amount_ml = 2

        # logic to promote the animal to the second training stage:
        is_animal_in_hardest_stage = any(
            item in self.df.trial_type.unique() for item in ["left_hard", "right_hard"]
        )
        if total_sessions >= 2 and not is_animal_in_hardest_stage:
            last_session_performance = self.get_session_performance(total_sessions)
            previous_session_performance = self.get_session_performance(
                total_sessions - 1
            )
            if (
                last_session_performance >= 0.85
                and previous_session_performance >= 0.85
            ):
                # introduce punishment
                self.settings.punishment = True
                # change the trial types
                self.settings.trial_types = [
                    "left_easy",
                    "right_easy",
                    "left_hard",
                    "right_hard",
                ]

        return None
    
    def define_gui_tabs(self) -> None:
        """
        This method is called when the GUI is created.
        It is used to define the organization of the settings in the GUI.
        Whatever that is not defined here will be placed in the "General" tab.

        You can use the 'Hide' tab to hide a setting from the GUI.

        Items in the lists need to have the same name as your settings variables.
        
        You can also restrict the possible values for each setting.
        """
        self.gui_tabs = {
            "Training": ["timer_for_response", "middle_port_light_intensity", "side_port_light_intensities"],
            "Dummies": ["g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"],
            "Hide": ["a", "b", "c", "d", "e", "f"],
        }

        # define possible values for each variable
        self.gui_tabs_restricted = {
            "timer_for_response": [0, .5, 1, 5, 10],
            "reward_amount_ml": [1, 2, 3, 4, 5],
        }


    def get_session_performance(self, session: int) -> float:
        """
        This method calculates the performance of a session.
        """

        return self.df[self.df.session == session].correct.mean()


# for debugging purposes
if __name__ == "__main__":
    import random

    import pandas as pd

    training = TrainingSettings()
    dfdir = "/home/pi/Downloads/B15.csv"
    training.df = pd.read_csv(dfdir, sep=";")
    training.update_training_settings()

    # # create a new column of randomly picked boolean values
    # training.df["correct"] = [random.choice([True, False]) for _ in range(training.df.shape[0])]
    # # save it
    # training.df.to_csv(dfdir, sep=";", index=False)
