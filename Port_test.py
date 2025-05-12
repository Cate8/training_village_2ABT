import random

from village.classes.task import Event, Output, Task


class Port_test(Task):

    def __init__(self):
        super().__init__()

        self.info = """
Testing ports, photogates and pumps
----------------------------------------------------------------
it's designed to test valves, leds and photogates and to charge 
water in the tubes if it's needed.
"""

    def start(self):
        self.side = random.choice(["left", "right"])

        # pumps
        if self.system_name == "9":
            self.valve_l_time = self.water_calibration.get_valve_time(port = 2, volume = self.settings.volume)
            self.valve_r_time = self.water_calibration.get_valve_time(port = 5, volume = self.settings.volume)
        elif self.system_name == "12":
            self.valve_l_time = self.water_calibration.get_valve_time(port = 1, volume = self.settings.volume)
            self.valve_r_time = self.water_calibration.get_valve_time(port = 7, volume = self.settings.volume)
           
        # counters
        self.trial_count = 0
        # to keep track of the number of trials on the same side
        self.same_side_count = self.settings.trials_with_same_side 
        self.reward_drunk = 0

        # Valve time,Valve reward, LED lights, pokes definition
        if self.system_name == "9":
            self.valve_l_time = self.water_calibration.get_valve_time(port = 2, volume = self.settings.volume)
            self.valve_l_reward = Output.Valve2

            self.valve_r_time = self.water_calibration.get_valve_time(port = 5, volume = self.settings.volume)
            self.valve_r_reward = Output.Valve5

            self.LED_l_on = (Output.PWM2, self.settings.led_intensity)
            self.LED_c_on = (Output.PWM3, self.settings.led_intensity)
            self.LED_r_on = (Output.PWM5, self.settings.led_intensity)

            self.left_poke_in = Event.Port2In 
            self.center_poke_in = Event.Port3In
            self.right_poke_in = Event.Port5In 
            self.left_poke_out = Event.Port2In 
            self.center_poke_out = Event.Port3In
            self.right_poke_out = Event.Port5In 

        elif self.system_name == "12":
            self.valve_l_time = self.water_calibration.get_valve_time(port = 1, volume = self.settings.volume)
            self.valve_r_time = self.water_calibration.get_valve_time(port = 7, volume = self.settings.volume)
           
            self.valve_l_reward = Output.Valve1
            self.valve_r_reward = Output.Valve7 

            self.LED_l_on = (Output.PWM1, self.settings.led_intensity)
            self.LED_c_on = (Output.PWM4, self.settings.led_intensity)
            self.LED_r_on = (Output.PWM7, self.settings.led_intensity)

            self.left_poke_in = Event.Port1In 
            self.center_poke_in = Event.Port4In
            self.right_poke_in = Event.Port7In 
            self.left_poke_out = Event.Port1In 
            self.center_poke_out = Event.Port4In
            self.right_poke_out = Event.Port7In 

        #### CREATING STATE MACHINE, ADDING STATES, SENDING AND RUNNING ####
        
    def create_trial(self):
        
        self.bpod.add_state(
            state_name='waiting',
            state_timer=10,
            state_change_conditions={Event.Tup: 'exit', 
                                      self.left_poke_in: 'left',
                                      self.right_poke_in: 'right'
                                    },
            output_actions=[]
            )

        self.bpod.add_state(
            state_name='left',
            state_timer = 0,
            state_change_conditions={self.left_poke_out: 'waiting',
                                     },
            output_actions=[self.LED_l_on, self.valve_l_reward]
            )
        
        self.bpod.add_state(
            state_name='center',
            state_timer = 0,
            state_change_conditions={self.center_poke_out: 'waiting',
                                     },
            output_actions=[self.LED_c_on]
            )

        self.bpod.add_state(
            state_name='right',
            state_timer = 0,
            state_change_conditions={self.right_poke_out: 'waiting'},
            output_actions=[self.LED_r_on, self.valve_r_reward])
    

    def after_trial(self):
        pass

    def close(self):
        pass
    



