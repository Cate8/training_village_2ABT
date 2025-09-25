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

        self.initial_side = random.choice(["left", "right"])
        self.initial_parity = 0 if self.initial_side == "left" else 1

        self.trial_count = 0
        self.reward_drunk = 0

        self.ports = BpodPorts(
            n_box=self.system_name,
            water_calibration=self.water_calibration,
            sound_calibration=self.sound_calibration,
            settings=self.settings
        )
        
    def create_trial(self):
        
        idx0 = self.current_trial - 1

        block_len = int(self.settings.trials_with_same_side)  # es. 30
        block_idx = idx0 // block_len                         # 0,0,...,0,1,1,...,1,2,...
        parity = (block_idx + self.initial_parity) % 2        

        self.side = "left" if parity == 0 else "right"

        # --- mapping ---
        if self.side == "left":
            self.correct_side = "left"
            self.wrong_side   = "right"
            self.correct_poke = self.ports.left_poke
            self.wrong_poke   = self.ports.right_poke
            self.valvetime    = self.ports.valve_l_time
            self.valve_action = self.ports.valve_l_reward
        else:
            self.correct_side = "right"
            self.wrong_side   = "left"
            self.correct_poke = self.ports.right_poke
            self.wrong_poke   = self.ports.left_poke
            self.valvetime    = self.ports.valve_r_time
            self.valve_action = self.ports.valve_r_reward

        print(self.valvetime)
        print(self.valve_action)


        #### CREATING STATE MACHINE, ADDING STATES, SENDING AND RUNNING ####
        
        # --- state machine (unchanged, ma paga solo un lato perché water_delivery usa self.valve_action) ---
        self.bpod.add_state(
            state_name='c_led_on',
            state_timer=self.settings.c_led_on_time,
            state_change_conditions={Event.Tup: 'drink_delay',
                                    self.ports.center_poke: 'side_led_on'},
            output_actions=[self.ports.LED_c_on]
        )

        self.bpod.add_state(
            state_name='side_led_on',
            state_timer=self.settings.led_on_time,
            state_change_conditions={Event.Tup: 'drink_delay',
                                    self.correct_poke: 'water_delivery',
                                    self.wrong_poke: 'penalty'},
            output_actions=[self.ports.LED_l_on, self.ports.LED_r_on]
        )

        self.bpod.add_state(
            state_name='water_delivery',
            state_timer=self.valvetime,
            state_change_conditions={Event.Tup: 'drink_delay'},
            output_actions=[self.valve_action]   # ← paga SOLO il lato selezionato
        )

        self.bpod.add_state(
            state_name='penalty',
            state_timer=self.settings.penalty_time,
            state_change_conditions={Event.Tup: 'drink_delay'},
            output_actions=[Output.SoftCode1]
        )

        self.bpod.add_state(
            state_name='drink_delay',
            state_timer=self.settings.drink_delay_time,
            state_change_conditions={Event.Tup: 'exit'},
            output_actions=[]
        )

    def after_trial(self):
        # salva il target del trial (lato corretto mostrato)
        self.register_value('rewarded_side', self.side)


        first_poke = self.trial_data.get('STATE_side_led_on_START')
        if first_poke and len(first_poke) > 0 and first_poke[0] > 0:
            water = 0
            outcome = "miss"
            response_side = "none"

            state = self.trial_data.get('STATE_water_delivery_START')
            if state and len(state) > 0 and state[0] > 0:
                water = self.settings.volume
                outcome = "correct"
                response_side = self.correct_side
                
            state = self.trial_data.get('STATE_penalty_START')
            if state and len(state) > 0 and state[0] > 0:
                outcome = "incorrect"
                response_side = self.wrong_side
        else:
            water = 0
            outcome = "omission"
            response_side = "none"


        # registra
        self.register_value('water', water)
        self.register_value('outcome', outcome)
        self.register_value('response_side', response_side)

    def close(self):
        pass
        
        




        
        



