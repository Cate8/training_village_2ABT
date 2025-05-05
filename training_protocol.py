from village.classes.training import Training


class TrainingProtocol(Training):
    """
    This class defines how the training protocol is going to be.
    This is, how variables change depending on different conditions (e.g. performance),
    and/or which tasks are going to be run.

    In this class 2 methods need to be implemented:
    - __init__
    - default_training_settings
    - update_training_settings

    In default_training_settings all the variables that can modify the state of 
    the training protocol must be defined.
    In update_training_settings the variables are updated depeding on the
    performance of the animal.
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
        self.settings.next_task = "Habituation"
        self.settings.refractary_period = 3600 * 4
        self.settings.minimum_duration = 16 * 60
        self.settings.maximum_duration = 20 * 60

        # Settings in this block are dependent on each task,
        # and the user needs to create and define them here
        self.settings.volume = 7 # ul of water delivered
        self.settings.led_intensity = 255 # led intensity (it's at maximum)
        self.settings.trials_with_same_side = 0 # number of trials with the same side
        self.settings.penalty_time = 0  # used from S3, time to wait after the wrong poke
        self.settings.led_on_time =  5 * 60 # side led on in S1 and S2
        self.settings.c_led_on_time = 5 * 60 # centre led on in S3
        self.settings.drink_delay_time = 5 # used in all tasks, time to wait after the reward is delivered
        self.settings.penalty_time = 0 # from S3,time to wait after the wrong poke

        """
        The following variables are used in S4 tasks, they'll vary along the
        vriation to progressively increase the difficulty of the 2ABT 
        and they are defined s follows:
            - N_trials: max number of trials in the session
            - N_blocks: max number of blocks in the session
            - mean_x: mean trial duration in the block  
            - block_type: can be 'fixed' (always mean_x trial in block)
            and 'exp'(exponential distribution with mean_x trials as a mean  )
            - prob_right_values: probability usend in the blocks during the session
            if you want the prob_Right to be ONLY 0.8 and 0.2, 
            then make this list prob_right_values = [0.8]
            - prob_block_type: can be 'rdm_values' or 'permutation_prob_list'
            'rdm_values' means that the prob_Right in Right blocks is randomly selected from the
            prob_right_values list. The prob_Left in Left blocks is 1 - prob_Right
            'permutation_prob_list' means that the prob_Right in Right blocks is selected from the
            prob_right_values list, but the order of the blocks is permuted.
            - prob_Left_Right_blocks: can be 'balanced' meaning that
            the prob_Right in Right blocks is the same as the prob_Left on Left blocls
            It can also be independent meaning that the prob_Right in Right blocks is
            INDEP of the prob_Left in Left blocks.
            This can cause that in some sessions, the overall prob_R over
            the entire session is larger than 0.5
            - lambda_param is the mean of the ITI distribution, (1/5 so 2 seconds)

        """ 
        self.settings.N_trials = 1000
        self.settings.prob_right_values = [0.9]  
        self.settings.N_blocks = 100
        self.settings.mean_x = 30
        self.settings.block_type = "fixed" 
        self.settings.prob_block_type = 'permutation_prob_list'
        self.settings.prob_Left_Right_blocks = 'balanced'
        self.settings.lambda_param = 0.5 #2 seconds

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

        if self.last_task == "S_hab":
            sessions_in_S_hab = self.df[self.df.task == "S_hab"].copy()
            if sessions_in_S_hab >= 2:
                self.settings.next_task = "S1"
                self.settings.minimum_duration = 20 * 60
                self.settings.maximum_duration = 45 * 60
                self.settings.trials_with_same_side = 20
                self.settings.drink_delay_time = 5


        if self.last_task == "S1":
            sessions_in_S1 = self.df[self.df.task == "S1"].copy()
            n_sessions_in_S1 = sessions_in_S1[sessions_in_S1.trial >= 3].session.count()
            if n_sessions_in_S1 >= 2:
                self.settings.next_task = "S2"
                self.settings.minimum_duration = 30 * 60
                self.settings.maximum_duration = 45 * 60
                self.settings.volume = 5
                self.settings.trials_with_same_side = 20
                self.settings.led_on_time = 300
                self.settings.drink_delay_time = 5

        if self.last_task == "S2":
            sessions_in_S2 = self.df[self.df.task == "S2"].copy()
            n_sessions_in_S2 = sessions_in_S2[sessions_in_S2.trial >= 3].session.count()
            if n_sessions_in_S2 >= 2:
                self.settings.next_task = "S3"
                self.settings.minimum_duration = 30 * 60
                self.settings.maximum_duration = 45 * 60
                self.settings.volume = 3
                self.settings.trials_with_same_side = 20
                self.settings.led_on_time = 300
                self.settings.drink_delay_time = 5
                self.settings.led_on_time = 5 * 60 
                self.settings.c_led_on_time = 5 * 60 

                self.settings.penalty_time = 0


        if self.last_task == "S3":
            sessions_in_S3 = self.df[self.df.task == "S3"].copy()
            n_sessions_in_S3 = sessions_in_S3[sessions_in_S3.trial >= 3].session.count()
            if n_sessions_in_S3 >= 3:
                self.settings.next_task = "S4_0"
                self.settings.minimum_duration = 30 * 60
                self.settings.maximum_duration = 45 * 60
                self.settings.volume = 3
                self.settings.trials_with_same_side = 30
                self.settings.drink_delay_time = 5
                self.settings.led_on_time = 5 * 60 
                self.settings.c_led_on_time = 5 * 60 
                self.settings.penalty_time = 0

                self.settings.prob_right_values = [0.9]  
                self.settings.block_type = "fixed" 
                self.settings.prob_block_type = 'permutation_prob_list'
                self.settings.prob_Left_Right_blocks = 'balanced'
                self.settings.lambda_param = 0.5 #2 seconds
                
        if self.last_task == "S4_0":
            sessions_in_S4_0 = self.df[self.df.task == "S4_0"].copy()
            n_sessions_in_S4_0 = sessions_in_S4_0[sessions_in_S4_0.trial >= 3].session.count()
            if n_sessions_in_S4_0 >= 3:
                self.settings.next_task = "S4_1"
                self.settings.minimum_duration = 30 * 60
                self.settings.maximum_duration = 45 * 60
                self.settings.volume = 3
                self.settings.trials_with_same_side = 30
                self.settings.c_led_on_time = 5 * 60 
                self.settings.led_on_time = 5 * 60 
                self.settings.penalty_time = 0
                self.settings.drink_delay_time = 5

                self.settings.prob_right_values = [0.9, 0.8]  
                self.settings.block_type = "fixed" 
                self.settings.prob_block_type = 'permutation_prob_list'
                self.settings.prob_Left_Right_blocks = 'balanced'
                self.settings.lambda_param = 0.25 #3 seconds

        if self.last_task == "S4_1":
            sessions_in_S4_1 = self.df[self.df.task == "S4_1"].copy()
            n_sessions_in_S4_1 = sessions_in_S4_1[sessions_in_S4_1.trial >= 3].session.count()
            if n_sessions_in_S4_1 >= 3:
                self.settings.next_task = "S4_2"
                self.settings.minimum_duration = 30 * 60
                self.settings.maximum_duration = 45 * 60
                self.settings.volume = 3
                self.settings.trials_with_same_side = 30
                self.settings.c_led_on_time = 5 * 60 
                self.settings.led_on_time = 5 * 60 
                self.settings.penalty_time = 0
                self.settings.drink_delay_time = 5

                self.settings.prob_right_values = [0.9, 0.8, 0.7]  
                self.settings.block_type = "fixed" 
                self.settings.prob_block_type = 'permutation_prob_list'
                self.settings.prob_Left_Right_blocks = 'balanced'
                self.settings.lambda_param = 0.2 #5 seconds

        if self.last_task == "S4_2":
            sessions_in_S4_2 = self.df[self.df.task == "S4_2"].copy()
            n_sessions_in_S4_2 = sessions_in_S4_2[sessions_in_S4_2.trial >= 3].session.count()
            if n_sessions_in_S4_2 >= 3:
                self.settings.next_task = "S4_3"
                self.settings.minimum_duration = 30 * 60
                self.settings.maximum_duration = 45 * 60
                self.settings.volume = 3
                self.settings.trials_with_same_side = 30
                self.settings.c_led_on_time = 5 * 60 
                self.settings.led_on_time = 5 * 60 
                self.settings.penalty_time = 0
                self.settings.drink_delay_time = 5

                self.settings.prob_right_values = [0.9, 0.8, 0.7]  
                self.settings.block_type = "exp" 
                self.settings.prob_block_type = 'permutation_prob_list'
                self.settings.prob_Left_Right_blocks = 'balanced'
                self.settings.lambda_param = 0.2 #5 seconds

        if self.last_task == "S4_3":
            self.settings.next_task = "S4_3"


                

            
   
