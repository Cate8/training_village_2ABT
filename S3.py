import random

from village.classes.task import Event, Output, Task
from BpodPorts import BpodPorts


class S3(Task):

    def __init__(self):
        super().__init__()

        self.info = """
Center-Initiated Side Alternation Task
-----------------------------------------------------------------------------------
Purpose: Tech mice o fixat in the center port, Assess discrimination, side bias, 
and behavioral flexibility in mice.
• Structure:
    - Mice initiate each trial by poking the center port (center LED on).
    - After center poke, both side LEDs turn on.
    - Only one side delivers reward (correct side); the other is inactive.
    - Reward side remains constant for 30 trials, then switches (left ↔ right).
    - Initial side is randomly selected at session start.
• Trial logic:
    - Correct poke → water delivery.
    - Wrong poke → timeout penalty (no reward).
    - After reward or timeout, short delay before next trial.
"""

    def start(self):

        self.side = random.choice(["left", "right"])
        # counters
        self.trial_count = 0
        # to keep track of the number of trials on the same side
        self.same_side_count = self.settings.trials_with_same_side 
        self.reward_drunk = 0

        self.ports = BpodPorts(
            n_box=self.system_name,
            water_calibration=self.water_calibration,
            sound_calibration=self.sound_calibration,
            settings=self.settings
        )

    def create_trial(self):

        if self.same_side_count == 0:
            # change side and reset the counter 
            self.side = "left" if self.side == "right" else "right"
            self.same_side_count = self.settings.trials_with_same_side
 
        self.same_side_count -= 1
        self.trial_count += 1
            
        # Correct and wrong choices definition
        if self.side == "left":
            self.correct_side = self.side
            self.wrong_side = "right"
            self.correct_poke = self.ports.left_poke
            self.wrong_poke = self.ports.right_poke
            self.valvetime = self.ports.valve_l_time
            self.valve_action = self.ports.valve_l_reward
        else:
            self.correct_side = self.side
            self.wrong_side = "left"
            self.correct_poke = self.ports.right_poke
            self.wrong_poke = self.ports.left_poke
            self.valvetime = self.ports.valve_r_time
            self.valve_action = self.ports.valve_r_reward
        
      

        print(self.side)
        print(self.valvetime)
        print(self.valve_action)



        #### CREATING STATE MACHINE, ADDING STATES, SENDING AND RUNNING ####
        
        print('')
        print('Trial: ' + str(self.current_trial))
        print('Reward side: ' + str(self.side))

        self.bpod.add_state(
            state_name='c_led_on',
            state_timer= self.settings.c_led_on_time,
            state_change_conditions={Event.Tup: 'drink_delay',
                                    self.ports.center_poke: 'side_led_on'},
            output_actions=[self.ports.LED_c_on]
            )

        self.bpod.add_state(
            state_name='side_led_on',
            state_timer= self.settings.led_on_time,
            state_change_conditions={Event.Tup: 'drink_delay', 
                                    self.correct_poke: 'water_delivery',
                                    self.wrong_poke: 'penalty'
                                    },

            output_actions=[self.ports.LED_l_on, self.ports.LED_r_on]
            )

        self.bpod.add_state(
            state_name='water_delivery',
            state_timer= self.valvetime,
            state_change_conditions={Event.Tup: 'drink_delay'},
            output_actions=[self.valve_action]
            )
        
        self.bpod.add_state(
            state_name='penalty',
            state_timer= self.settings.penalty_time,
            state_change_conditions={Event.Tup: 'drink_delay'},
            output_actions=[Output.SoftCode1]
            )


        self.bpod.add_state(
            state_name='drink_delay',
            state_timer= self.settings.drink_delay_time ,
            state_change_conditions={Event.Tup: 'exit'},
            output_actions=[])

    def after_trial(self):
        # Relevant prints
        print(self.trial_data)

        self.register_value('side', self.side)

        # register how much water was delivered
        water_delivered = self.trial_data.get("STATE_water_delivery_START", False)
        if water_delivered:
            self.register_value('water', self.settings.volume)
        else:
            self.register_value('water', 0)

        # get the outcome of the trial
        if 'STATE_water_delivery_START' in self.current_trial_states and len(self.current_trial_states['STATE_water_delivery_START']) > 0:
            water_delivery_start = self.current_trial_states['STATE_water_delivery_START'][0]

            if water_delivery_start > 0:
                self.outcome = "correct"
        elif 'STATE_penalty_START' in self.current_trial_states and len(self.current_trial_states['STATE_penalty_START']) > 0:
            wrong_side = self.current_trial_states['STATE_penalty_START'][0]

            if wrong_side > 0:
                self.outcome = "incorrect"
        elif 'STATE_side_LED_on_START' in self.current_trial_states and len(self.current_trial_states['STATE_side_LED_on_START']) > 0:
            side_light_start = self.current_trial_states['STATE_side_LED_on_START'][0]

            if side_light_start > 0:
                self.outcome = "miss"
        else:

            self.outcome = "omission"

    def close(self):
        pass
        
        



