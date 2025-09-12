import random

from village.classes.task import Event, Output, Task
import random
import numpy as np
from scipy.signal import firwin, lfilter 
import warnings
from utils_functions import (generate_uniform_block_duration, 
                             generate_block_probs_vec, 
                             generate_block_reward_side_vec, 
                             custom_random_iti)



class S4_3(Task):

    def __init__(self):
        super().__init__()

        self.info = """
HARD S4: Probabilistic Two-Armed Bandit with Blocked Reward Contingencies
---------------------------------------------------------------------------------------
This task implements a version of the probabilistic two-alternative forced choice (2AFC) task using dynamically changing reward probabilities structured in blocks.
• Mice initiate each trial with a center poke, followed by a choice between l or r ports.
• Reward is delivered probabilistically based on the block-specific probability of reward on the right port (pR), while the left port's probability is 1-pR.
• The task alternates through a sequence of blocks, each with:
- A certain number of trials (drawn from a uniform or fixed distribution)
- A reward probability for the right port (either fixed, random, or permuted from a list)
• The side rewarded on each trial is drawn from a binomial distribution with p = pR.
• Inter-trial intervals (ITIs) are generated from a truncated exponential distribution.
----------------------------------------------------------------------------------------
Task Variables:  
PROBABILITIES = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]  
BLOCK TYPE = VARIABLE LENGHT (MIN 20 MAX 55)
MEAN ITI DISTRIBUTION = 5 seconds UP TO 30 seconds
"""
    def start(self):

        self.side = random.choice(["left", "right"])
        # counters
        self.trial_count = 0
        # to keep track of the number of trials on the same side
        self.same_side_count = self.settings.trials_with_same_side 
        self.reward_drunk = 0

        # Valve time,Valve reward, LED lights, pokes definition 
        # according to the system name 
        if self.system_name == "9":
            self.valve_l_time = self.water_calibration.get_valve_time(port = 2, volume = self.settings.volume)
            self.valve_l_reward = Output.Valve2

            self.valve_r_time = self.water_calibration.get_valve_time(port = 5, volume = self.settings.volume)
            self.valve_r_reward = Output.Valve5

            self.LED_l_on = (Output.PWM2, self.settings.led_intensity)
            self.LED_c_on = (Output.PWM3, self.settings.led_intensity)
            self.LED_r_on = (Output.PWM5, self.settings.led_intensity)

            self.left_poke = Event.Port2In 
            self.center_poke = Event.Port3In
            self.right_poke = Event.Port5In 

        elif self.system_name == "12":
            self.valve_l_time = self.water_calibration.get_valve_time(port = 7, volume = self.settings.volume)
            self.valve_r_time = self.water_calibration.get_valve_time(port = 1, volume = self.settings.volume)
           
            self.valve_l_reward = Output.Valve7
            self.valve_r_reward = Output.Valve1 

            self.LED_l_on = (Output.PWM7, self.settings.led_intensity)
            self.LED_c_on = (Output.PWM4, self.settings.led_intensity)
            self.LED_r_on = (Output.PWM1, self.settings.led_intensity)

            self.left_poke = Event.Port7In 
            self.center_poke = Event.Port4In
            self.right_poke = Event.Port1In 
    

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
        #print("probs_vector: ", self.probs_vector)
        #print("reward_side_vec_fixed_prob: ", self.reward_side_vec_fixed_prob)
        #print("Tailored ITI values: ", self.random_iti_values)

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
            
        # Correct and wrong choices definition
        if self.reward_side_number == 0:  # 0  for reward on the left side
            self.correct_side = "left"
            self.wrong_side = "right"
            self.correct_side = self.side
            self.correct_poke = self.left_poke
            self.wrong_poke = self.right_poke
            self.valvetime = self.valve_l_time
            self.valve_action = self.valve_l_reward

        else:  # 1 for reward on the right side
            self.correct_side = self.side
            self.wrong_side = "left"
            self.correct_poke = self.right_poke
            self.wrong_poke = self.left_poke
            self.valvetime = self.valve_r_time
            self.valve_action = self.valve_r_reward

        #### CREATING STATE MACHINE, ADDING STATES, SENDING AND RUNNING ####
        
        print('')
        print('Trial: ' + str(self.current_trial))
        print('Reward side: ' + str(self.side))

        self.bpod.add_state(
            state_name='c_led_on',
            state_timer= self.settings.c_led_on_time,
            state_change_conditions={Event.Tup: 'drink_delay',
                                    self.center_poke: 'side_led_on'},
            output_actions=[self.LED_c_on]
            )

        self.bpod.add_state(
            state_name='side_led_on',
            state_timer= self.settings.led_on_time,
            state_change_conditions={Event.Tup: 'drink_delay', 
                                    self.correct_poke: 'water_delivery',
                                    self.wrong_poke: 'penalty'
                                    },

            output_actions=[self.LED_l_on, self.LED_r_on]
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
        # Relevant prints
        print(self.trial_data)

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
        elif 'STATE_wrong_side' in self.current_trial_states and len(self.current_trial_states['STATE_wrong_side']) > 0:
            wrong_side = self.current_trial_states['STATE_wrong_side'][0]

            if wrong_side > 0:
                self.outcome = "incorrect"
        elif 'STATE_side_LED_on_START' in self.current_trial_states and len(self.current_trial_states['STATE_side_LED_on_START']) > 0:
            side_light_start = self.current_trial_states['STATE_side_LED_on_START'][0]

            if side_light_start > 0:
                self.outcome = "miss"
        else:

            self.outcome = "omission"

        self.register_value('side', self.side)
        self.register_value('correct_poke', self.correct_poke)
        self.register_value('probability_r', self.probability)
        self.register_value('Block_index', self.block_identity)
        self.register_value('Block_type', self.settings.block_type)
        self.register_value('Prob_block_type', self.settings.prob_block_type)
        self.register_value('Probability_L_R_blocks', self.settings.prob_Left_Right_blocks)
        self.register_value('list_prob_R_values', self.settings.prob_right_values)
        self.register_value('outcome', self.outcome)
        self.register_value('iti_duration', self.random_iti)

    def close(self):
        pass
        
        



