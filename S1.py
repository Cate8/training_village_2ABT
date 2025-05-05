import random

from village.classes.task import Event, Output, Task


class S1(Task):

    def __init__(self):
        super().__init__()

        self.info = """
        ########   TASK INFO   ########
        -----------------------------------------------------------------------------
        Task S1 – Water Delivery Task
        -----------------------------------------------------------------------------
        Passive learning, at the moment of the lick, the reward is already delivered. 

        - Each trial starts with:
            * Reward valve opens (water is delivered)
            * LED on the rewarded port turns on
      

        - The LED remains ON until:
            * A poke is detected in the correct port
            * Or a timeout occurs

        - First trial delivers water automatically and waits for poke.
        -----------------------------------------------------------------------------
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

    def create_trial(self):

        if self.same_side_count == 0:
            # change side and reset the counter 
            self.side = "left" if self.side == "right" else "right"
            self.same_side_count = self.settings.trials_with_same_side
 
        self.same_side_count -= 1
        self.trial_count += 1
            
        if self.system_name == "9":
            if self.side == "left":
                self.valvetime = self.valve_l_time
                self.valve_action = Output.Valve2
                self.light_LED = (Output.PWM2, self.settings.led_intensity)
                self.poke_LED_on_side= Event.Port2In
                self.poke_LED_off_side= Event.Port5In

            else:
                self.valvetime = self.valve_r_time
                self.valve_action = Output.Valve5
                self.light_LED = (Output.PWM5, self.settings.led_intensity)
                self.poke_LED_on_side= Event.Port5In   
                self.poke_LED_off_side= Event.Port2In
        
        elif self.system_name == "12":  
            if self.side == "left":
                self.valvetime = self.valve_l_time
                self.valve_action = Output.Valve7
                self.light_LED = (Output.PWM7, self.settings.led_intensity)
                self.poke_LED_on_side= Event.Port7In
                self.poke_LED_off_side= Event.Port1In

            else:
                self.valvetime = self.valve_r_time
                self.valve_action = Output.Valve1
                self.light_LED = (Output.PWM1, self.settings.led_intensity)
                self.poke_LED_on_side= Event.Port1In
                self.poke_LED_off_side= Event.Port7In 

        print(self.system_name)
        print(self.side)
        print(self.valvetime)
        print(self.valve_action)
        print(self.light_LED)


        #### CREATING STATE MACHINE, ADDING STATES, SENDING AND RUNNING ####
        
        print('')
        print('Trial: ' + str(self.current_trial))
        print('Reward side: ' + str(self.side))

        self.bpod.add_state(
            state_name='water_delivery',
            state_timer=self.valvetime,
            state_change_conditions={Event.Tup: 'led_on'},
            output_actions=[self.light_LED, self.valve_action]
            )

        self.bpod.add_state(
            state_name='led_on',
            state_timer = self.settings.led_on_time ,
            state_change_conditions={Event.Tup: 'drink_delay',
                                     self.poke_LED_on_side: 'drink_delay',
                                     self.poke_LED_off_side: 'register_off_poke'},
            output_actions=[self.light_LED]
            )

        self.bpod.add_state(
            state_name='register_off_poke',
            state_timer = 0 ,
            state_change_conditions={Event.Tup: 'led_on'},
            output_actions=[]
            )

        self.bpod.add_state(
            state_name='drink_delay',
            state_timer = self.settings.drink_delay_time,
            state_change_conditions={Event.Tup: 'exit'},
            output_actions=[])



    def after_trial(self):
        # Relevant prints

        print(self.trial_data)
        self.register_value('side', self.side)
        self.register_value('water', self.settings.volume)
    


    def close(self):
        pass
    



