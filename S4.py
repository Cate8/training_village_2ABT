import random

from village.classes.task import Event, Output, Task
from village.manager import manager
from village.settings import settings 



class S3(Task):

    def __init__(self):
        super().__init__()

        self.info = """
        ########   TASK INFO   ########
------------------------------
 Task S3 – Center-Initiated Side Alternation Task
------------------------------
• Purpose: Assess discrimination, side bias, and behavioral flexibility in mice.
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
• Hardware support:
    - Compatible with Box 9 and Box 12 configurations.
    - Adapts valve, LED, and port mappings accordingly.
• Outputs & data:
    - Side of reward, total water consumed, and poke responses tracked per trial.
      ------------------------------
        ########   PORTS INFO, BOX 9  ########
        Port 1 - Right port
        Port 3 - Central port
        Port 5 - Left port
        ########   PORTS INFO, BOX 9  ########
        Port 7 - Right port
        Port 3 - Central port
        Port 1 - Left port
        """

    def start(self):
        self.trials_with_same_side = 30
        self.side = random.choice(["left", "right"])
        self.volume = 7
        self.penalty_time = 0

        # counters
        self.trial_count = 0
        # to keep track of the number of trials on the same side
        self.same_side_count = self.trials_with_same_side 
        self.reward_drunk = 0
        self.led_intensity = 255

        # Valve time,Valve reward, LED lights, pokes definition 
        # according to the system name 
        if settings.get("SYSTEM_NAME") == "9":
            self.valve_l_time = manager.water_calibration.get_valve_time(port = 2, volume = self.volume)
            self.valve_l_reward = Output.Valve2

            self.valve_r_time = manager.water_calibration.get_valve_time(port = 5, volume = self.volume)
            self.valve_r_reward = Output.Valve5

            self.LED_l_on = (Output.PWM2, self.led_intensity)
            self.LED_c_on = (Output.PWM3, self.led_intensity)
            self.LED_r_on = (Output.PWM5, self.led_intensity)

            self.left_poke = Event.Port2In 
            self.center_poke = Event.Port3In
            self.right_poke = Event.Port5In 

        elif settings.get("SYSTEM_NAME") == "12":
            self.valve_l_time = manager.water_calibration.get_valve_time(port = 1, volume = self.volume)
            self.valve_r_time = manager.water_calibration.get_valve_time(port = 7, volume = self.volume)
           
            self.valve_l_reward = Output.Valve1
            self.valve_r_reward = Output.Valve7 

            self.LED_l_on = (Output.PWM1, self.led_intensity)
            self.LED_c_on = (Output.PWM4, self.led_intensity)
            self.LED_r_on = (Output.PWM7, self.led_intensity)

            self.left_poke = Event.Port1In 
            self.center_poke = Event.Port4In
            self.right_poke = Event.Port7In 
            



    def create_trial(self):

        if self.same_side_count == 0:
            # change side and reset the counter 
            self.side = "left" if self.side == "right" else "right"
            self.same_side_count = self.trials_with_same_side
 
        self.same_side_count -= 1
        self.trial_count += 1
            
        # Correct and wrong choices definition
        if self.side == "left":
            self.correct_side = self.side
            self.wrong_side = "right"
            self.correct_poke = self.left_poke
            self.wrong_poke = self.right_poke
            self.valvetime = self.valve_l_time
            self.valve_action = self.valve_l_reward


        else:
            self.correct_side = self.side
            self.wrong_side = "left"
            self.correct_poke = self.right_poke
            self.wrong_poke = self.left_poke
            self.valvetime = self.valve_r_time
            self.valve_action = self.valve_r_reward
        
      

        print(settings.get("SYSTEM_NAME"))
        print(self.side)
        print(self.valvetime)
        print(self.valve_action)



        #### CREATING STATE MACHINE, ADDING STATES, SENDING AND RUNNING ####
        
        print('')
        print('Trial: ' + str(self.current_trial))
        print('Reward side: ' + str(self.side))

        self.bpod.add_state(
            state_name='c_led_on',
            state_timer= 300,
            state_change_conditions={Event.Tup: 'drink_delay',
                                    self.center_poke: 'side_LED_on'},
            output_actions=[self.LED_c_on]
            )

        self.bpod.add_state(
            state_name='side_LED_on',
            state_timer= 300,
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
            state_timer= self.penalty_time,
            state_change_conditions={Event.Tup: 'drink_delay'},
            output_actions=[Output.SoftCode1]
            )


        self.bpod.add_state(
            state_name='drink_delay',
            state_timer= 5,
            state_change_conditions={Event.Tup: 'exit'},
            output_actions=[])



    def after_trial(self):
        self.reward_drunk += self.volume

        # Relevant prints
        self.register_value('side', self.side)
        self.register_value('reward_drunk', self.reward_drunk)



