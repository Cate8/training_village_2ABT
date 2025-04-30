from village.classes.training import Training


class TrainingProtocol(Training):
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
    following the logic in the update method."""

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
        self.settings.next_task = "Simple"
        self.settings.refractary_period = 10
        self.settings.minimum_duration = 60
        self.settings.maximum_duration = 120

        # Settings in this block are dependent on each task,
        # and the user needs to create and define them here
        self.settings.reward_amount_ml = 0.05

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
        return
