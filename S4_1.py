import random

from village.classes.task import Event, Output, Task
import random
import numpy as np
from scipy.signal import firwin, lfilter 
import warnings
from utils_functions import (generate_uniform_block_duration, 
                             generate_block_probs_vec, 
                             generate_block_reward_side_vec, 
                             custom_random_iti,
                             )

from BpodPorts import BpodPorts
import time

class S4_1(Task):

    def __init__(self):
        super().__init__()

        self.info = """
EASY-MEDIUM S4: Probabilistic Two-Armed Bandit with Blocked Reward Contingencies
---------------------------------------------------------------------------------------
This task implements a version of the probabilistic two-alternative forced choice (2AFC) task using dynamically changing reward probabilities structured in blocks.
• Mice initiate each trial with a center poke, followed by a choice between l or r ports.
• Reward is delivered probabilistically based on the block-specific probability of reward on the right port (pR), while the left port's probability is 1-pR.
• The task alternates through a sequence of blocks, each with:
- A certain number of trials (drawn from a uniform or fixed distribution)
- A reward probability for the right port (either fixed, random, or permuted from a list)
• The side rewarded on each trial is drawn from a binomial distribution with p = pR.
• Inter-trial intervals (ITIs) are generated from a truncated exponential distribution.
-----------------------------------------------------------------------------------
Task Variables:  
PROBABILITIES = [0.9, 0.1, 0.8, 0.2]  
BLOCK TYPE = exponential,from 20 and 55 trials
MEAN ITI DISTRIBUTION = 3 seconds UP TO 30 seconds
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

        #Generate the vector with the block duration (in trials)  for each block
        self.block_duration_vec = generate_uniform_block_duration(x_type= self.settings.block_type, 
                                                                  mean_x= self.settings.mean_x, 
                                                                  N_blocks=self.settings.N_blocks
                                                                  )

        #Generate the vector with the p_Right values for each block
        self.probs_vector = generate_block_probs_vec(self.settings.N_blocks, 
                                                     self.settings.prob_block_type , 
                                                     self.settings.prob_right_values, 
                                                     self.settings.prob_Left_Right_blocks
                                                     )

        #Generate the binary vector with the Right (1) and Left (0) rewarded sides in each trial
        self.reward_side_vec_fixed_prob = generate_block_reward_side_vec(self.settings.N_blocks,
                                                                        self.block_duration_vec, 
                                                                        self.probs_vector
                                                                        )

        #Generate the vector tailored ITIs values (from 1 to 30 sec, final mean=5 sec)
        self.random_iti_values = custom_random_iti(self.settings.N_trials, 1, self.settings.lambda_param)

        print("block_duration_vec: ", self.block_duration_vec)
        print("probs_vector: ", self.probs_vector)
        print("reward_side_vec_fixed_prob: ", self.reward_side_vec_fixed_prob)
        print("Tailored ITI values: ", self.random_iti_values)

    def create_trial(self):

        # current_trial starts at 1 we want to start at 0
        self.probability = self.reward_side_vec_fixed_prob[self.current_trial-1][0]
        self.reward_side_number = self.reward_side_vec_fixed_prob[self.current_trial-1][1]
        self.block_identity = self.reward_side_vec_fixed_prob[self.current_trial-1][2]
        self.random_iti = self.random_iti_values[self.current_trial-1]


        print("current_trial: ", self.current_trial)
        print("block_identity: ", self.block_identity)
        print("probability: ", self.probability)
        print("reward_side_number: ", self.reward_side_number)
        #print("ITI_duration: ", self.random_iti)
            
        if self.reward_side_number == 0:  # left
            self.correct_side = "left"
            self.wrong_side = "right"
            self.correct_poke = self.ports.left_poke
            self.wrong_poke = self.ports.right_poke
            self.valvetime = self.ports.valve_l_time
            self.valve_action = self.ports.valve_l_reward

        else:  # right
            self.correct_side = "right"
            self.wrong_side = "left"
            self.correct_poke = self.ports.right_poke
            self.wrong_poke = self.ports.left_poke
            self.valvetime = self.ports.valve_r_time
            self.valve_action = self.ports.valve_r_reward

        #### CREATING STATE MACHINE, ADDING STATES, SENDING AND RUNNING ####
        
        print('')
        print('Trial: ' + str(self.current_trial))
        print('Reward side: ' + str(self.correct_side))

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
            state_timer= self.random_iti,
            state_change_conditions={Event.Tup: 'exit'},
            output_actions=[])


    def after_trial(self):
  
        # --- LOCAL helpers :  ---
        def _event_key(ev):
            return ev.name if hasattr(ev, "name") else str(ev)

        def _first_choice_after(t0, left_key: str, right_key: str):
            """Ritorna ('left'|'right', t_choice) del primo poke dopo t0, altrimenti (None, None)."""
            L = [t for t in self.trial_data.get(left_key,  []) if t >= t0]
            R = [t for t in self.trial_data.get(right_key, []) if t >= t0]
            tL = L[0] if L else None
            tR = R[0] if R else None
            if tL is None and tR is None:
                return None, None
            if tL is None:
                return "right", tR
            if tR is None:
                return "left", tL
            return ("left", tL) if tL <= tR else ("right", tR)

        # 1) timestamp side LED ON (start)
        side_on_key = "STATE_side_led_on_START"
        t_side_on = self.trial_data[side_on_key][0] if side_on_key in self.trial_data and self.trial_data[side_on_key] else None

        # 2) events keys poke L/R
        left_key  = _event_key(self.ports.left_poke)   # es. "Port5In" / "Port2In" / ...
        right_key = _event_key(self.ports.right_poke)  # es. "Port1In" / "Port5In" / ...

        # 3) frist choice after LED on
        if t_side_on is not None:
            first_resp, t_choice = _first_choice_after(t_side_on, left_key, right_key)
        else:
            first_resp, t_choice = (None, None)

        # 4) Rewarded trial (0=left, 1=right)
        rewarded_side = "right" if int(self.reward_side_number) == 1 else "left"

        # 5) outcome  CHOSEN SIDE -> 'side'
        if first_resp is None:
            chosen_side = "none"
            correct_outcome_int = 0
            outcome  = "miss"
            water = 0
        else:
            chosen_side = first_resp
            correct_outcome_int = int(chosen_side == rewarded_side)
            outcome = "correct" if correct_outcome_int else "incorrect"
            water = self.settings.volume if correct_outcome_int else 0


        first_poke = self.trial_data.get('STATE_side_led_on_START')
        if first_poke and len(first_poke) > 0 and first_poke[0] > 0:
            pass
        else:
            outcome = "omission"


        # --- REGISTRATION ---
        if t_choice is not None:
            self.register_value('first_trial_response_time', t_choice)

        self.register_value('water', water)
        self.register_value('correct_poke', self.correct_poke)
        self.register_value('probability_r', self.probability)
        self.register_value('Block_index', self.block_identity)
        self.register_value('Block_type', self.settings.block_type)
        self.register_value('Prob_block_type', self.settings.prob_block_type)
        self.register_value('Probability_L_R_blocks', self.settings.prob_Left_Right_blocks)
        self.register_value('list_prob_R_values', self.settings.prob_right_values)
        self.register_value('outcome', outcome) # 'correct', 'incorrect', 'miss'
        self.register_value("rewarded_side", self.correct_side) # side that was rewarded this trial
        self.register_value("response_side", chosen_side) # side the animal chose
        self.register_value('iti_duration', self.random_iti)

        print("registration")
        print(f"  Rewarded side: {rewarded_side}")
        print(f"  Response side: {first_resp}")
        print(f"  Outcome: {outcome}")
        print(f"  Probability right: {self.probability}")
        print("")


    def close(self):
        pass
        
        



