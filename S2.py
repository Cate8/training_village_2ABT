import random

from village.classes.task import Event, Output, Task



class S2(Task):

    def __init__(self):
        super().__init__()

        self.info = """
        ########   TASK INFO   ########
        -----------------------------------------------------------------------------
        Task S2 – Water Delivery Task variation
        -----------------------------------------------------------------------------
        - The task is designed to teach mice to approach the lickport.
        Active learning, at the moment of the lick, the reward will be delivered. 

        - Each trial starts with:
            * LED on the rewarded port turns on (one of the two ports)
            * The animal as to poke in the port with the led on
            * Reward valve opens (water is delivered)
      

        - The LED remains ON until:
            * A poke is detected in the correct port
            * Or a timeout occurs
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
                self.poke_side= Event.Port2In

            else:
                self.valvetime = self.valve_r_time
                self.valve_action = Output.Valve5
                self.light_LED = (Output.PWM5, self.settings.led_intensity)
                self.poke_side= Event.Port5In   
        
        elif self.system_name == "12":  
            if self.side == "left":
                self.valvetime = self.valve_l_time
                self.valve_action = Output.Valve7
                self.light_LED = (Output.PWM7, self.settings.led_intensity)
                self.poke_side= Event.Port7In

            else:
                self.valvetime = self.valve_r_time
                self.valve_action = Output.Valve1
                self.light_LED = (Output.PWM1, self.settings.led_intensity)
                self.poke_side= Event.Port1In   

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


