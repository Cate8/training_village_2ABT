import random

from village.classes.task import Event, Output, Task
from BpodPorts import BpodPorts

import time

class S2(Task):

    def __init__(self):
        super().__init__()

        self.info = """
Active learning, Water Delivery Task variation
-----------------------------------------------------------------------------
The task is designed to teach mice to approach the lickport:
- Each trial starts with:
    * LED on the rewarded port turns on (one of the two ports)
    * The animal as to poke in the port with the led on
    * Reward valve opens (water is delivered)
- The LED remains ON until:
    * A poke is detected in the correct port
    * Or a timeout occurs
If the animal pokes in the wrong port nothing will happen, the mice 
will remain in the same state until pokes in the "correct" port. The same 
reward side will be repeated for n trials (20).
"""

    def start(self):
        self.side = random.choice(["left", "right"])

        self.ports = BpodPorts(
            n_box=self.system_name,
            water_calibration=self.water_calibration,
            sound_calibration=self.sound_calibration,
            settings=self.settings
        )

        # counters
        self.trial_count = 0
        # to keep track of the number of trials on the same side
        self.same_side_count = self.settings.trials_with_same_side 
        self.reward_drunk = 0

    def create_trial(self):
        if self.same_side_count == 0:
            # change side and reset the counter 
            self.side = "left" if self.side == "right" else "right"
            self.same_side_count = self.settings.trials_with_same_side
 
        self.same_side_count -= 1
        self.trial_count += 1

        if self.side == "left":
            self.valvetime = self.ports.valve_l_time
            self.valve_action = self.ports.valve_l_reward
            self.light_LED = self.ports.LED_l_on
            self.poke_side= self.ports.left_poke
        else:
            self.valvetime = self.ports.valve_r_time
            self.valve_action = self.ports.valve_r_reward
            self.light_LED = self.ports.LED_r_on
            self.poke_side= self.ports.right_poke


        print(self.side)
        print(self.valvetime)
        print(self.valve_action)
        print(self.light_LED)
        print(self.poke_side)


        #### CREATING STATE MACHINE, ADDING STATES, SENDING AND RUNNING ####
        
        print('')
        print('Trial: ' + str(self.current_trial))
        print('Reward side: ' + str(self.side))

        self.bpod.add_state(
            state_name='led_on',
            state_timer= 300,
            state_change_conditions={Event.Tup: 'drink_delay', self.poke_side: 'water_delivery'},
            output_actions=[self.light_LED]
            )

        self.bpod.add_state(
            state_name='water_delivery',
            state_timer=self.valvetime,
            state_change_conditions={Event.Tup: 'drink_delay'},
            output_actions=[self.valve_action, self.light_LED]
            )

        self.bpod.add_state(
            state_name='drink_delay',
            state_timer= 5,
            state_change_conditions={Event.Tup: 'exit'},
            output_actions=[])



    def after_trial(self):
        # Relevant prints
        self.register_value('side', self.side)
        
        # register how much water was delivered
        water_delivered = self.trial_data.get("STATE_water_delivery", False)
        if water_delivered:
            self.register_value('water', self.settings.volume)
        else:
            self.register_value('water', 0)

        # get the outcome of the trial
        if 'STATE_water_delivery_START' in self.current_trial_states and len(self.current_trial_states['STATE_water_delivery_START']) > 0:
            water_delivery_start = self.current_trial_states['STATE_water_delivery_START'][0]

            if water_delivery_start > 0:
                self.outcome = "poke_action"
    
        elif 'STATE_side_LED_on_START' in self.current_trial_states and len(self.current_trial_states['STATE_side_LED_on_START']) > 0:
            side_light_start = self.current_trial_states['STATE_side_LED_on_START'][0]

            if side_light_start > 0:
                self.outcome = "poke_missed"
 

    def close(self):
        pass


