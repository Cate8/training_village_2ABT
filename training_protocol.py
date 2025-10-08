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
    - self.refractory_period
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
        self.settings.next_task = "S0"
        self.settings.refractory_period = 240 * 60
        self.settings.minimum_duration = 10 * 60
        self.settings.maximum_duration =  15 * 60

        # Settings in this block are dependent on each task,
        # and the user needs to create and define them here
        
        #GENERAL SETTINGS
        self.settings.volume = 7 # ul of water delivered
        self.settings.led_intensity = 255 # led intensity (it's at maximum)

        #SHAPING SETTINGS:S1 AND S2
        self.settings.trials_with_same_side = 20 # number of trials with the same side
        self.settings.led_on_time =  5 * 60 # side led on in S1 and S2
        self.settings.penalty_time = 0  # used from S3, time to wait after the wrong poke
        self.settings.drink_delay_time = 5 # used in all shaping task, 
        #time to wait after the reward is delivered
        #SHAPING SETTINGS:S3
        self.settings.c_led_on_time = 5 * 60 # centre led on in S3
        self.settings.penalty_time = 0 # from S3,time to wait after the wrong poke

        """
        TASK SETTINGS:Two-Armed Bandit Task (2ABT) – Detailed Description 
        (S4 AND it's variations)
        --------------------------------------------------------------------------
        This task is a probabilistic two-alternative forced choice (2AFC) paradigm designed 
        to assess value-based decision-making and flexibility in mice.
        Mice initiate each trial by poking a center port, which triggers the illumination 
        of both left and right side ports. The animal must choose between these two options,
        but only one port is probabilistically rewarded on each trial. The probability of
        reward associated with each side changes dynamically across blocks of trials, 
        requiring the animal to continuously track and update its internal estimate of which
        side is currently more rewarding.

        Block Structure & Reward Probabilities
        ---------------------------------------
        The session is divided into multiple blocks (e.g., 100), each lasting a variable 
        number of trials. The length of each block is drawn from a uniform distribution.

        Each block is assigned a reward probability for the right port (pR):
        - In "Right" blocks: pR is high (e.g., 0.9), left = 1 - pR.
        - In "Left" blocks: pR is low (e.g., 0.1), meaning the left is more rewarding.

        The sequence of blocks alternates between left- and right-favored, with the option 
        for:
        - Balanced mode: pR for left blocks is always 1 - pR from the paired right block.
        - Independent mode: pR values are randomly assigned, possibly creating imbalances 
        over the session.
        --------------------------------------------------------------------------
        The following variables are used in S4 tasks, they are defined as follows:
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
            - lambda_param is the mean of the ITI distribution, (if it's 
              0.5 = 1/5 so 2 seconds)
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
        if self.last_task == "S0":
            df_S0 = self.df[self.df["task"] == "S0"]
            if len(df_S0) >= 1:
                self.settings.next_task = "S1"
                self.settings.minimum_duration = 25 * 60
                self.settings.maximum_duration = 45 * 60
                self.settings.trials_with_same_side = 20
                self.settings.drink_delay_time = 5
            else:
                self.settings.next_task = "S0"
        

        elif self.last_task == "S1":
            df_S1 = self.df[self.df["task"] == "S1"]
            if len(df_S1) >= 2:
                df_last_two_session_s1 = df_S1.iloc[-2:]
                n_trials_S1 = df_last_two_session_s1.trial.sum()
                if n_trials_S1 >= 100:
                    self.settings.next_task = "S2"
                    self.settings.minimum_duration = 25 * 60
                    self.settings.maximum_duration = 45 * 60
                    self.settings.volume = 3
                    self.settings.trials_with_same_side = 20
                    self.settings.led_on_time = 300
                    self.settings.drink_delay_time = 2
                else:
                    self.settings.next_task = "S1" 
            else:
                self.settings.next_task = "S1" # Keep in task until it meets the criteria

        elif self.last_task == "S2":
            df_S2 = self.df[self.df.task == "S2"]
            if len(df_S2) >= 2:
                df_last_two_session_S2 = df_S2.iloc[-2:]
                n_trials_S2 = df_last_two_session_S2.trial.sum()
                if n_trials_S2 >= 100:
                    self.settings.next_task = "S3"
                    self.settings.minimum_duration = 30 * 60
                    self.settings.maximum_duration = 45 * 60
                    self.settings.volume = 3
                    self.settings.trials_with_same_side = 30
                    self.settings.led_on_time = 300
                    self.settings.drink_delay_time = 2
                    self.settings.led_on_time = 5 * 60 
                    self.settings.c_led_on_time = 5 * 60 

                    self.settings.penalty_time = 0
                else:
                    self.settings.next_task = "S2" 
            else:
                self.settings.next_task = "S2" # Keep in task until it meets the criteria

        elif self.last_task == "S3":
            df_S3 = self.df[self.df.task == "S3"]
            if len(df_S3) >= 3:
                df_last_two_session_S3 = df_S3.iloc[-2:]
                n_trials_S3 = df_last_two_session_S3.trial.sum()
                if n_trials_S3 >= 200:
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
                else:
                    self.settings.next_task = "S3" 
            else:
                self.settings.next_task = "S3" # Keep in task until it meets the criteria
                
        elif self.last_task == "S4_0":
            df_S4_0 = self.df[self.df.task == "S4_0"]
            if len(df_S4_0) >= 3:
                df_last_two_session_S4_0 = df_S4_0.iloc[-2:]
                n_trials_S4_0 = df_last_two_session_S4_0.trial.sum()
                if n_trials_S4_0 >= 250:
                    self.settings.next_task = "S4_0"
                    self.settings.minimum_duration = 45 * 60
                    self.settings.maximum_duration = 50 * 60
                    self.settings.volume = 2
                    self.settings.trials_with_same_side = 30 
                    self.settings.c_led_on_time = 5 * 60 
                    self.settings.led_on_time = 5 * 60 
                    self.settings.penalty_time = 0
                    self.settings.drink_delay_time = 5

                    self.settings.prob_right_values = [0.9]  
                    self.settings.block_type = "fixed" 
                    self.settings.prob_block_type = 'permutation_prob_list'
                    self.settings.prob_Left_Right_blocks = 'balanced'
                    self.settings.lambda_param = 0.3 #3,3 seconds
                else:
                    self.settings.next_task = "S4_0" 
            else:
                self.settings.next_task = "S4_0" # Keep in task until it meets the criteria
        
        if self.last_task == "S4_0":
            self.settings.next_task = "S4_0"

        # elif self.last_task == "S4_1":
        #     df_S4_1 = self.df[self.df.task == "S4_1"]
        #     if len(df_S4_1) >= 3:
        #         df_last_two_session_S4_1 = df_S4_1.iloc[-2:]
        #         n_trials_S4_1 = df_last_two_session_S4_1.trial.sum()
        #         if n_trials_S4_1 >= 300:
        #             self.settings.next_task = "S4_2"
        #             self.settings.minimum_duration = 30 * 60
        #             self.settings.maximum_duration = 45 * 60
        #             self.settings.volume = 2
        #             self.settings.trials_with_same_side = 30
        #             self.settings.c_led_on_time = 5 * 60 
        #             self.settings.led_on_time = 5 * 60 
        #             self.settings.penalty_time = 0
        #             self.settings.drink_delay_time = 5

        #             self.settings.prob_right_values = [0.9, 0.8, 0.7]  
        #             self.settings.block_type = "fixed" 
        #             self.settings.prob_block_type = 'permutation_prob_list'
        #             self.settings.prob_Left_Right_blocks = 'balanced'
        #             self.settings.lambda_param = 0.25 #4 seconds
        #         else:
        #             self.settings.next_task = "S4_1" 
        #     else:
        #         self.settings.next_task = "S4_1" # Keep in this task until it meets the criteria

        
        # elif self.last_task == "S4_2":
        #     df_S4_2 = self.df[self.df.task == "S4_2"]
        #     if len(df_S4_2) >= 3:
        #         df_last_two_session_S4_2 = df_S4_2.iloc[-2:]
        #         n_trials_S4_2 = df_last_two_session_S4_2.trial.sum()
        #         if n_trials_S4_2 >= 300:
        #             self.settings.next_task = "S4_3"
        #             self.settings.minimum_duration = 30 * 60
        #             self.settings.maximum_duration = 45 * 60
        #             self.settings.volume = 2
        #             self.settings.trials_with_same_side = 30
        #             self.settings.c_led_on_time = 5 * 60 
        #             self.settings.led_on_time = 5 * 60 
        #             self.settings.penalty_time = 0
        #             self.settings.drink_delay_time = 5

        #             self.settings.prob_right_values = [0.9, 0.8, 0.7, 0.6]  
        #             self.settings.block_type = "exp" 
        #             self.settings.prob_block_type = 'permutation_prob_list'
        #             self.settings.prob_Left_Right_blocks = 'balanced'
        #             self.settings.lambda_param = 0.2 #5 seconds
        #         else:
        #             self.settings.next_task = "S4_2" 
        #     else:
        #         self.settings.next_task = "S4_2" # Keep in this task until it meets the criteria

        # if self.last_task == "S4_3":
        #     self.settings.next_task = "S4_3"


                

            
   
