import random

from village.classes.task import Event, Output, Task
from village.manager import manager


class FollowTheLight(Task):
    def __init__(self):
        super().__init__()

        self.info = """

        Follow The Light Task
        -------------------

        This task is a simple visual task where the mouse has
        to poke the center port to start a trial.
        After the center port is poked,
        one of the two side ports will be illuminated.
        If the mouse licks the correct side port, it receives a reward.
        If the mouse licks the wrong side port, it receives a punishment.

        It contains 2 training stages:
        - Training stage 1: Only one side port is illuminated and gives reward.
                            No punishment is given, and the mouse can choose again.
        - Training stage 2: Both ports are illuminated with different intensity.
                            Brighter port gives reward, the other one gives punishment.
        """

        # variables are defined in training_settings.py

    def start(self):

        print("FollowTheLight starts")

        ## Initiate states that won't change during training
        # Trial start state:
        # Turn on light in the middle port
        self.ready_to_initiate_output = [
            (
                Output.PWM2,
                self.settings.middle_port_light_intensity,
            )
        ]

        # Time the valve needs to open to deliver the reward amount
        # Make sure to calibrate the valve before using it, otherwise this function
        # will return the default value of 0.01 seconds
        self.left_valve_opening_time = manager.water_calibration.get_valve_time(
            port=1, volume=self.settings.reward_amount_ml
        )
        self.right_valve_opening_time = manager.water_calibration.get_valve_time(
            port=3, volume=self.settings.reward_amount_ml
        )

        # determine if punishment will be used
        if self.settings.punishment:
            self.punish_condition = "punish_state"
        else:
            # if no punishment is used, let the mouse choose again
            self.punish_condition = "stimulus_state"

    def configure_gui(self):
        # TODO: implement this method
        pass

    def create_trial(self):
        """
        This function updates the variables that will be used every trial
        """
        print("")
        print("Trial {0}".format(str(self.current_trial)))

        ## Start the task
        # On the first trial, the entry door to the behavioral box gets closed.
        # This is coded as a transition in the 'close_door' state.
        if self.current_trial == 0:
            # Close the door
            self.start_of_trial_transition = "close_door"
        else:
            self.start_of_trial_transition = "ready_to_initiate"

        ## Define the conditions for the trial
        # pick a trial type at random.
        self.this_trial_type = random.choice(self.settings.trial_types)

        ## Set the variables for the stimulus states and the possible choices
        self.stimulus_state_output = []
        match self.this_trial_type:
            case "both_ports_on":
                self.light_intensity_left = self.settings.side_port_light_intensities[2]
                self.light_intensity_right = self.settings.side_port_light_intensities[2]
                self.left_poke_action = "reward_state"
                # This is a bit shitty as I have to create two reward states
                # TODO: Create a habituation task to exemplify that posibility also
            case "left_easy":
                self.stimulus_state_output.append(
                    (
                        Output.PWM1,
                        self.task_settings["side_port_light_intensity"],
                    )
                )
                self.left_poke_action = "reward_state"
                self.valve_opening_time = self.left_valve_opening_time
                self.right_poke_action = self.punish_condition
                self.valve_to_open = Output.Valve1
            case "right_easy":
                self.stimulus_state_output.append(
                    (
                        Output.PWM3,
                        self.task_settings["side_port_light_intensity"],
                    )
                )
                self.left_poke_action = self.punish_condition
                self.right_poke_action = "reward_state"
                self.valve_opening_time = self.right_valve_opening_time
                self.valve_to_open = Output.Valve4

        # assemble the state machine
        self.assemble_state_machine()

    def assemble_state_machine(self):
        # 'start_of_trial' state that sends a TTL pulse from the BNC channel 2
        # This can be used to synchronize the task with other devices (not used here)
        self.bpod.add_state(
            state_name="start_of_trial",
            state_timer=0.001,
            state_change_conditions={Event.Tup: self.start_of_trial_transition},
            output_actions=[Output.BNC2High],
        )

        self.bpod.add_state(
            state_name="close_door",
            state_timer=0,
            state_change_conditions={Event.Tup: "ready_to_initiate"},
            output_actions=[Output.SoftCode20],
            # TODO: change this softcode to a default one
        )

        # 'ready_to_initiate' state that waits for the poke in the middle port
        self.bpod.add_state(
            state_name="ready_to_initiate",
            state_timer=0,
            state_change_conditions={Event.Port2In: "stimulus_state"},
            output_actions=self.ready_to_initiate_output,
        )

        self.bpod.add_state(
            state_name="stimulus_state",
            state_timer=self.settings.timer_for_response,
            state_change_conditions={
                Event.Port1In: self.left_poke_action,
                Event.Port3In: self.right_poke_action,
                Event.Tup: "exit",
            },
            output_actions=[
                (Output.PWM1,self.light_intensity_left),
                (Output.PWM3,self.light_intensity_right),
            ],
        )

        self.bpod.add_state(
            state_name="reward_state",
            state_timer=self.valve_opening_time,
            state_change_conditions={Event.Tup: "iti"},
            output_actions=[self.valve_to_open],
        )

        self.bpod.add_state(
            state_name="punish_state",
            state_timer=self.settings.punishment_time,
            state_change_conditions={Event.Tup: "iti"},
            output_actions=[],
        )

        # iti is the time that the mouse has to wait before the next trial
        self.bpod.add_state(
            state_name="iti",
            state_timer=self.settings.iti,
            state_change_conditions={Event.Tup: "exit"},
            output_actions=[],
        )

    def after_trial(self):
        pass


    def close(self):
        print("Closing the task")
