import random

from village.classes.task import Event, Output, Task
from village.manager import manager
from village.settings import settings 



class S2(Task):

    def __init__(self):
        super().__init__()

        self.info = """
        ########   TASK INFO   ########
        This task teaches mice to approach the lickport. Active learning, at the moment
        of the lick, the reward will be delivered. 

        - Each trial starts with:
            * LED on the rewarded port turns on (one of the two ports)
            * The animal as to poke in the port with the led on
            * Reward valve opens (water is delivered)
      

        - The LED remains ON until:
            * A poke is detected in the correct port
            * Or a timeout occurs


        Global light stays ON throughout the session.

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
        self.trials_with_same_side = 20
        self.side = random.choice(["left", "right"])
        self.volume = 7

        # pumps
        if settings.get("SYSTEM_NAME") == "9":
            self.valve_l_time = manager.water_calibration.get_valve_time(port = 2, volume = self.volume)
            self.valve_r_time = manager.water_calibration.get_valve_time(port = 5, volume = self.volume)
        elif settings.get("SYSTEM_NAME") == "12":
            self.valve_l_time = manager.water_calibration.get_valve_time(port = 1, volume = self.volume)
            self.valve_r_time = manager.water_calibration.get_valve_time(port = 7, volume = self.volume)

        # counters
        self.trial_count = 0
        # to keep track of the number of trials on the same side
        self.same_side_count = self.trials_with_same_side 
        self.reward_drunk = 0
        self.led_intensity = 255

    def create_trial(self):

        if self.same_side_count == 0:
            # change side and reset the counter 
            self.side = "left" if self.side == "right" else "right"
            self.same_side_count = self.trials_with_same_side
 
        self.same_side_count -= 1
        self.trial_count += 1
            

        if settings.get("SYSTEM_NAME") == "9":
            if self.side == "left":
                self.valvetime = self.valve_l_time
                self.valve_action = Output.Valve2
                self.light_LED = (Output.PWM2, self.led_intensity)
                self.poke_side= Event.Port2In

            else:
                self.valvetime = self.valve_r_time
                self.valve_action = Output.Valve5
                self.light_LED = (Output.PWM5, self.led_intensity)
                self.poke_side= Event.Port5In   
        
        elif settings.get("SYSTEM_NAME") == "12":  
            if self.side == "left":
                self.valvetime = self.valve_l_time
                self.valve_action = Output.Valve7
                self.light_LED = (Output.PWM7, self.led_intensity)
                self.poke_side= Event.Port7In

            else:
                self.valvetime = self.valve_r_time
                self.valve_action = Output.Valve1
                self.light_LED = (Output.PWM1, self.led_intensity)
                self.poke_side= Event.Port1In   

        print(settings.get("SYSTEM_NAME"))
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
        self.reward_drunk += self.volume

        # Relevant prints
        self.register_value('side', self.side)
        self.register_value('reward_drunk', self.reward_drunk)



