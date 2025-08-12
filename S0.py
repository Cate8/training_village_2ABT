from village.classes.task import Event, Output, Task
from village.manager import manager
from BpodPorts import BpodPorts

# click on the link below to see the documentation about how to create
# tasks, plots and training protocols
# https://braincircuitsbehaviorlab.github.io/village/user_guide/create.html


class S0(Task):
    def __init__(self):
        super().__init__()

        self.info = """
Habituation Task
----------------------------------------------------------
This task is an automatic mouse habitution to the box.
nothing will happen during the task, the mouse will be
left alone in the box for 15 minutes. Eventual pokes will 
be registered but no reward will be delivered. 
"""

    def start(self):
        """
        This function is called when the task starts.
        It is used to calculate values needed for the task.
        The following variables are accesible by default:
        - self.bpod: Bpod object
        - self.settings: Settings object
        - self.manager: Manager object
        - self.subject: Subject object
        - self.task: Task object
        - self.task_name: Task name


        - self.bpod: (Bpod object)
        - self.name: (str) the name of the task
        self.subject: (str) the name of the subject
        self.current_trial: (int) the current trial number starting from 1
        self.current_trial_states: (list) information about the current trial
        self.system_name: (str) the name of the system as defined in the
                                tab settings of the GUI
        self.settings: (Settings object) the settings defined in training_settings.py
        self.trial_data: (dict) information about the current trial
        self.force_stop: (bool) if made true the task will stop
        self.maximum_number_of_trials: int = 100000000
        self.chrono = time_utils.Chrono()

        Al the variables created in training_settings.py are accessible here.
        """

        # In training_settins we created the following variables that are accesible here:

        # Time the valve needs to open to deliver the reward amount
        # Make sure to calibrate the valve before using it, otherwise this function
        # will return the default value of 0.01 seconds

        # self.left_valve_opening_time = manager.water_calibration.get_valve_time(
        #     port=1, volume=self.settings.reward_amount_ml
        # )
        # self.right_valve_opening_time = manager.water_calibration.get_valve_time(
        #     port=3, volume=self.settings.reward_amount_ml
        # )

        # # use maximum light intensity for both side ports
        # self.light_intensity_left = self.settings.side_port_light_intensities[-1]
        # self.light_intensity_right = self.settings.side_port_light_intensities[-1]
        self.ports = BpodPorts(
            n_box=self.system_name,
            water_calibration=self.water_calibration,
            settings=self.settings
        )

    def create_trial(self):
        """
        This function modifies variables and then sends the state machine to the bpod
        before each trial.
        """
        """
                if self.system_name == "9":
                    self.ports.left_poke= Event.Port2In
                    self.ports.center_poke= Event.Port3In
                    self.ports.right_poke= Event.Port5In


                elif self.system_name == "12": 
                    self.ports.left_poke= Event.Port7In
                    self.ports.center_poke= Event.Port5In
                    self.ports.right_poke= Event.Port1In 
        """

        # 'ready_to_initiate': state that turns on the middle port light and
        # waits for a poke in the central port (Port2)
       
        if self.current_trial == 1:

            self.bpod.add_state(
                state_name="trial_0",
                state_timer= 1,
                state_change_conditions={Event.Tup: "exit"},
                output_actions=[],
            )
    
        self.bpod.add_state(
            state_name="ready_to_explore",
            state_timer= 5 * 60,
            state_change_conditions={Event.Tup: "exit", 
                                     self.ports.left_poke: 'left_poke',
                                     self.ports.center_poke: 'center_poke',
                                     self.ports.right_poke: 'right_poke'},
            output_actions=[],
        )

        self.bpod.add_state(
            state_name="left_poke",
            state_timer= 0,
            state_change_conditions={Event.Tup: "exit"},
            output_actions=[],
        )

        self.bpod.add_state(
            state_name="center_poke",
            state_timer= 0,
            state_change_conditions={Event.Tup: "exit"},
            output_actions=[],
        )

        self.bpod.add_state(
            state_name="right_poke",
            state_timer= 0,
            state_change_conditions={Event.Tup: "exit"},
            output_actions=[],
        )

    def after_trial(self):
        """
        Here you can register all the values you need to save for each trial.
        It is essential to always include a variable named water, which stores the
        amount of water consumed during each trial.
        The system will calculate the total water consumption in each session
        by summing this variable.
        If the total water consumption falls below a certain threshold,
        an alarm will be triggered.
        This threshold can be adjusted in the Settings tab of the GUI.
        """
        if 'STATE_left_poke_START' in self.current_trial_states:
                self.outcome = "left_poke"

        if 'STATE_centre_poke_START' in self.current_trial_states:
                self.outcome = "center_poke"
        
        if 'STATE_right_poke_START' in self.current_trial_states:
                self.outcome = "right_poke"
    
        else: 
            self.outcome = "no_action"

        # Register the outcome of the trial
        self.register_value('poke_l', self.ports.left_poke)
        self.register_value('poke_c', self.ports.center_poke)
        self.register_value('poke_r', self.ports.right_poke)
        self.register_value('outcome', self.outcome)

    def close(self):
        """
        Here you can perform any actions you want to take once the task is completed,
        such as sending a message via email or Slack, creating a plot, and more.
        """

        print("closed!!")
