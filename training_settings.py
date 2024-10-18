from village.classes.training import Training


class TrainingSettings(Training):
    """In this class 2 methods need to be implemented:
    - __init__
    - update_training_settings

    In __init__ all the variables that can modify the state of the training protocol
    must be defined.
    When a new subject is created, a new row is added to the data/subjects.csv file.
    variables defined in __init__ and its values. The following variables are needed:
    - self.task
    - self.refractary_period
    - self.minimum_duration
    - self.maximum_duration
    - self.maximum_trials
    In addition to these variables, all the necessary variables to modify the state
    of the tasks can be included.

    When a task is run the values of the variables are read from the json file.
    When the task ends, the values of the variables are updated in the json file,
    following the logic in the update method"""

    def __init__(self) -> None:
        super().__init__()

    def default_training_settings(self) -> None:
        self.settings.next_task = "Simple"
        self.settings.refractary_period = 14400
        self.settings.minimum_duration = 1800
        self.settings.maximum_duration = 3600
        self.settings.maximum_number_of_trials = 10000
        self.settings.delay = 0.4
        self.settings.time = 3
        self.settings.side = "right"
        self.settings.mylist = [1, 2, 3]

    def update_training_settings(self) -> None:
        """This method is called every time a task finishes.
        Subject is the name of the subject (str).
        df is a pandas dataframe containing all the behavioral data from the subject."""

        return None
